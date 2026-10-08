"""Measure forced-choice credences from encoded answers and supplied logits.

Use CandidatePaths for each candidate's two encoded continuations, validate
them before inference, and request first_token_ids at the fixed context.
select_paths chooses one complete form per candidate; build_trie then tells
you which further scores are needed to distinguish the selected answers.
TriePlan.score returns their natural-log credences.

The caller owns tokenization and model calls. This module does no inference,
generation, or caching. Scores must be finite float32-range logits or -inf,
supplied as Python floats; probability calculations use higher precision.

>>> forms = {
...     "yes": CandidatePaths((10,), (30,)),
...     "no": CandidatePaths((20,), (40,)),
... }
>>> validate_potential_paths(forms)
>>> first_token_ids(forms)
(10, 20, 30, 40)
>>> initial = {10: 2.0, 30: 1.0, 20: 2.0, 40: 1.0}
>>> plan = build_trie(select_paths(forms, initial))
>>> logs = plan.score({(): initial})
>>> {label: round(math.exp(value), 3) for label, value in logs.items()}
{'yes': 0.5, 'no': 0.5}
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .errors import BackendReadoutError
from .probabilities import logprob_sum, masked_logprobs, validate_readout

type TokenPath = tuple[int, ...]


@dataclass(frozen=True, init=False)
class CandidatePaths:
    """Supply the two complete encoded continuations for one candidate.

    A token path is the sequence of model token IDs representing an answer
    after the fixed prompt context.
    ``bare`` represents the exact candidate text; ``spaced`` represents one
    additional ASCII space followed by that text. Whitespace changes
    tokenization: the two forms can have entirely different token IDs,
    lengths, and suffixes, not just different first tokens.

    The caller encodes both whole forms after the same fixed context and
    checks their decoded text. This class does not tokenize or change the
    context; existing context spaces stay in place. select_paths compares
    first-token logits to choose one complete form, preferring bare on ties.
    It does not maximize whole-answer likelihood.

    For illustration, a tokenizer might split ``"positive"`` into ``"pos"``,
    ``"it"``, and ``"ive"`` with IDs [973, 184, 1784], while encoding
    ``" positive"`` as one token [17384]. These are invented vocabulary entries:

    >>> forms = {
    ...     "positive": CandidatePaths((973, 184, 1784), (17384,)),
    ...     "negative": CandidatePaths((42, 43), (84,)),
    ... }
    >>> select_paths(forms, {973: 1.0, 17384: 2.0, 42: 2.0, 84: 1.0})
    {'positive': (17384,), 'negative': (42, 43)}
    """

    bare: TokenPath
    spaced: TokenPath

    def __init__(self, bare: Sequence[int], spaced: Sequence[int]) -> None:
        # Validate nonempty token-ID sequences and copy them so caller edits
        # cannot change the continuations held by this frozen object.
        object.__setattr__(self, "bare", _token_path(bare))
        object.__setattr__(self, "spaced", _token_path(spaced))


@dataclass(frozen=True)
class ReadoutRequest:
    """Describe which next-token logits to request and where to request them.

    Append ``prefix`` to the unchanged fixed context, then request every
    token ID in ``wanted`` there. Prefixes are continuation tokens, not full
    contexts. ``wanted`` may include stopping tokens (END) as well as answer
    tokens. Supply all requested scores to the plan, not only a top-K subset.

    Plans use ``prefix=()`` for a root branch, whose scores can be reused from
    the initial form-selection readout.

    >>> context = (7, 8)
    >>> request = ReadoutRequest(prefix=(10,), wanted=(11, 99, 100))
    >>> context + request.prefix, request.wanted
    ((7, 8, 10), (11, 99, 100))
    """

    prefix: TokenPath
    wanted: tuple[int, ...]


@dataclass(frozen=True)
class TriePlan:
    """An immutable readout plan for scoring already-selected answer paths.

    Obtain a plan with build_trie, supply the scores requested by ``readouts``,
    and call score for natural-log credences. ``paths`` retains the complete
    selected continuations in candidate order; ``end_tokens`` holds the
    stopping-token set (END).

    Readouts are in deterministic parent-first prefix order. Singleton
    transitions are forced with probability 1, and leaves stop without an
    END readout. Neither needs a query, but forced tokens remain in the
    prefixes used for later requests. Plans contain no context, scores, or
    model state and perform no inference or caching.

    >>> plan = build_trie({"up": (10,), "down": (20,)})
    >>> logs = plan.score({(): {10: 2.0, 20: 2.0}})
    >>> {label: round(math.exp(value), 3) for label, value in logs.items()}
    {'up': 0.5, 'down': 0.5}
    """

    paths: tuple[tuple[str, TokenPath], ...]
    end_tokens: tuple[int, ...]
    readouts: tuple[ReadoutRequest, ...]

    @property
    def additional_readouts(self) -> tuple[ReadoutRequest, ...]:
        """Return the requests still needed after the first-token readout.

        These are non-root branches only: request each at the fixed context
        plus its prefix. Reuse initial scores for any root request instead
        of querying the root again.

        >>> plan = build_trie(
        ...     {"up": (10,), "upward": (10, 11, 12), "down": (20,)},
        ...     end_tokens=(99, 100),
        ... )
        >>> context = (7, 8)
        >>> request, = plan.additional_readouts
        >>> context + request.prefix, request.wanted
        ((7, 8, 10), (11, 99, 100))
        """
        return tuple(request for request in self.readouts if request.prefix)

    def score(self, readouts: Mapping[TokenPath, Mapping[int, float]]) -> dict[str, float]:
        """Return a candidate-to-natural-log-credence mapping in candidate order.

        Supply ``readouts[prefix][token_id]`` for every request in the plan.
        Keys are continuation prefixes, not the full contexts used to query
        the model. Reuse initial form-selection scores under ``()`` if the
        selected root branches. Missing or invalid requested scores raise
        BackendReadoutError, even on zero-probability paths; extras are ignored.

        Each branch normalizes over its distinct allowed token IDs; all -inf
        scores give a uniform token distribution. Singleton steps are forced.
        Where one answer ends and another continues, probabilities of all END
        stopping tokens are summed for the shorter answer. Leaves need no END
        scores. The resulting credences sum to 1 without label-level
        renormalization. Exact zero credence is returned as -inf; finite logs
        remain useful even when exponentiation underflows.

        This uses supplied scores only, without inference or cached state.
        Here equal stopping and continuation logits favor stopping 2:1
        because END contains two distinct tokens:

        >>> plan = build_trie(
        ...     {"up": (10,), "upward": (10, 11, 12), "down": (20,)},
        ...     end_tokens=(99, 100),
        ... )
        >>> logs = plan.score({
        ...     (): {10: 2.0, 20: 2.0},
        ...     (10,): {11: 0.0, 99: 0.0, 100: 0.0},
        ... })
        >>> {label: round(math.exp(value), 3) for label, value in logs.items()}
        {'up': 0.333, 'upward': 0.167, 'down': 0.5}
        """
        transitions = {}
        for request in self.readouts:
            try:
                scores = readouts[request.prefix]
            except KeyError as error:
                raise BackendReadoutError(
                    f"Missing branch readout at prefix {request.prefix}"
                ) from error
            transitions[request.prefix] = masked_logprobs(request.wanted, scores)

        result = {}
        for label, path in self.paths:
            factors = []
            for request in self.readouts:
                prefix = request.prefix
                if path[: len(prefix)] != prefix:
                    continue
                if len(path) > len(prefix):
                    factors.append(transitions[prefix][path[len(prefix)]])
                elif path == prefix:
                    factors.append(
                        logprob_sum(transitions[prefix][token] for token in self.end_tokens)
                    )
            result[label] = math.fsum(factors)
        return result


def first_token_ids(forms: Mapping[str, CandidatePaths]) -> tuple[int, ...]:
    """Return the sorted token IDs to request for choosing candidate forms.

    Request these distinct first tokens once at the unchanged fixed context,
    then pass their logits to select_paths. Shared first tokens appear only
    once. This checks labels and path ownership, but use
    validate_potential_paths before inference to check stopping boundaries too.

    >>> forms = {
    ...     "up": CandidatePaths((10,), (30,)),
    ...     "upward": CandidatePaths((10, 11, 12), (31,)),
    ...     "down": CandidatePaths((20,), (40,)),
    ... }
    >>> first_token_ids(forms)  # Shared start 10 is requested only once.
    (10, 20, 30, 31, 40)
    """
    owners = _path_owners(_possible_paths(forms))
    return tuple(sorted({path[0] for path in owners}))


def select_paths(
    forms: Mapping[str, CandidatePaths], first_scores: Mapping[int, float]
) -> dict[str, TokenPath]:
    """Choose one complete encoded continuation per candidate for scoring.

    ``first_scores`` supplies logits for all first_token_ids at the same fixed
    context. The higher first-token logit wins; exact ties, including -inf
    ties, choose bare. The result preserves candidate order and the chosen
    form's entire suffix, even when the two forms have different lengths.

    This is not whole-answer-likelihood maximization and adds no probability
    factor to later trie scoring. All first-token scores must be present and
    valid, including those of discarded forms; otherwise BackendReadoutError
    is raised. The caller obtains scores; this function does no inference.

    >>> forms = {
    ...     "upward": CandidatePaths((10, 11, 12), (31,)),
    ...     "down": CandidatePaths((20,), (40,)),
    ... }
    >>> select_paths(forms, {10: 2.0, 31: 2.0, 20: 2.0, 40: 1.0})
    {'upward': (10, 11, 12), 'down': (20,)}
    >>> select_paths(forms, {10: 2.0, 31: 3.0, 20: 2.0, 40: 1.0})
    {'upward': (31,), 'down': (20,)}
    """
    scores = validate_readout(first_token_ids(forms), first_scores)
    return {
        label: paths.spaced if scores[paths.spaced[0]] > scores[paths.bare[0]] else paths.bare
        for label, paths in forms.items()
    }


def validate_potential_paths(
    forms: Mapping[str, CandidatePaths], *, end_tokens: Iterable[int] = ()
) -> None:
    """Check that either form can safely be selected before requesting logits.

    ``forms`` must contain at least two nonempty candidate labels. A complete
    token path cannot belong to different labels. If one label's possible
    path is a strict prefix of another's, ``end_tokens`` must supply stopping
    tokens (END), distinct from the next continuation token at that boundary.
    Otherwise stopping and continuing could not be distinguished.

    Invalid labels, collisions, or unusable stopping boundaries raise
    ValueError; valid inputs return None. Prefix relationships between one
    candidate's own forms need no END because only one form will be selected.
    This checks token paths, not whether they decode to the intended text.

    >>> forms = {
    ...     "up": CandidatePaths((10,), (30,)),
    ...     "upward": CandidatePaths((10, 11, 12), (31,)),
    ... }
    >>> validate_potential_paths(forms)
    Traceback (most recent call last):
        ...
    ValueError: Prefix (10,) needs a nonempty END set
    >>> validate_potential_paths(forms, end_tokens=(99, 100))
    """
    ends = _end_tokens(end_tokens)
    owners = _path_owners(_possible_paths(forms))
    for path, label in owners.items():
        for depth in range(1, len(path)):
            prefix = path[:depth]
            shorter_label = owners.get(prefix)
            if shorter_label is not None and shorter_label != label:
                _validate_stopping_boundary(prefix, (path[depth],), ends)


def build_trie(paths: Mapping[str, Sequence[int]], *, end_tokens: Iterable[int] = ()) -> TriePlan:
    """Plan the readouts needed to distinguish selected candidate answers.

    ``paths`` maps at least two nonempty candidate labels to one complete,
    nonempty encoded continuation each, all after the same fixed context.
    Use select_paths first when choosing between bare and spaced forms.
    Shared prefixes share probability mass rather than counting once per label.

    ``end_tokens`` supplies stopping tokens (END) when one answer is a strict
    prefix of another. At such a boundary END must be nonempty and disjoint
    from continuation IDs; every distinct stopping ID participates in scoring.
    Empty or cross-label identical paths and unusable stopping boundaries
    raise ValueError.

    The returned immutable TriePlan preserves candidate order and requests
    only branches. Singleton transitions have probability 1 without scores;
    leaves stop without END scores. Forced tokens still belong in the context
    plus prefix for later queries. Building a plan does no inference or caching.

    Here the only later request distinguishes stopping at ``"up"`` from
    continuing to ``"upward"``; the final token 12 needs no score:

    >>> plan = build_trie(
    ...     {"up": (10,), "upward": (10, 11, 12), "down": (20,)},
    ...     end_tokens=(99, 100),
    ... )
    >>> plan.additional_readouts
    (ReadoutRequest(prefix=(10,), wanted=(11, 99, 100)),)
    """
    _validate_labels(paths)
    selected = tuple((label, _token_path(path)) for label, path in paths.items())
    owners = _path_owners(selected)
    ends = _end_tokens(end_tokens)
    children: dict[TokenPath, set[int]] = {}
    for _, path in selected:
        for depth, token in enumerate(path):
            children.setdefault(path[:depth], set()).add(token)

    readouts = []
    for prefix, next_tokens in children.items():
        allowed = set(next_tokens)
        if prefix in owners:
            _validate_stopping_boundary(prefix, next_tokens, ends)
            allowed.update(ends)
        if len(allowed) > 1:
            readouts.append(ReadoutRequest(prefix, tuple(sorted(allowed))))
    readouts.sort(key=lambda request: (len(request.prefix), request.prefix))
    return TriePlan(paths=selected, end_tokens=ends, readouts=tuple(readouts))


def _possible_paths(forms: Mapping[str, CandidatePaths]) -> tuple[tuple[str, TokenPath], ...]:
    _validate_labels(forms)
    result = []
    for label, paths in forms.items():
        if not isinstance(paths, CandidatePaths):
            raise TypeError(f"Candidate {label!r} requires named bare and spaced paths")
        result.extend(((label, paths.bare), (label, paths.spaced)))
    return tuple(result)


def _path_owners(paths: Iterable[tuple[str, TokenPath]]) -> dict[TokenPath, str]:
    """Assign each path to one label, rejecting cross-label collisions."""
    owners = {}
    for label, path in paths:
        owner = owners.get(path)
        if owner is not None and owner != label:
            raise ValueError(f"Path {path} is owned by both {owner!r} and {label!r}")
        owners[path] = label
    return owners


def _validate_labels(paths: Mapping[str, object]) -> None:
    if len(paths) < 2:
        raise ValueError("At least two candidates are required")
    if any(not isinstance(label, str) or not label for label in paths):
        raise ValueError("Candidate labels must be nonempty strings")


def _token_path(values: Sequence[int]) -> TokenPath:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("A token path must be a sequence of integer token ids")
    path = _token_ids(values)
    if not path:
        raise ValueError("Token paths must be nonempty")
    return path


def _token_ids(values: Iterable[int]) -> tuple[int, ...]:
    tokens = tuple(values)
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
        raise ValueError("Token ids must be nonnegative integers")
    return tokens


def _end_tokens(values: Iterable[int]) -> tuple[int, ...]:
    """Return validated, sorted, distinct stopping-token IDs."""
    # Duplicate END IDs must not multiply stopping probability.
    return tuple(sorted(set(_token_ids(values))))


def _validate_stopping_boundary(
    prefix: TokenPath, continuations: Iterable[int], end_tokens: tuple[int, ...]
) -> None:
    """Require stopping tokens distinct from continuations at this prefix."""
    if not end_tokens:
        raise ValueError(f"Prefix {prefix} needs a nonempty END set")
    overlap = set(end_tokens).intersection(continuations)
    if overlap:
        raise ValueError(f"END overlaps continuation tokens {sorted(overlap)} at prefix {prefix}")
