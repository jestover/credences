"""Pure selection, path validation, and immutable token-trie readout plans.

The caller supplies already-encoded paths and score readouts. This module
does not tokenize text, query a model, generate answers, or retain scores.
Supported scores are finite float32-range logits or -inf, carried as Python
floats. Probability arithmetic and path accumulation use higher precision.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .errors import BackendReadoutError
from .probabilities import logprob_sum, masked_logprobs, validate_readout

type TokenPath = tuple[int, ...]


@dataclass(frozen=True, init=False)
class CandidatePaths:
    """The named bare and one-additional-space paths for a single candidate.

    Each is one complete canonical continuation, not a first-token variant.
    Both paths are nonempty and copied into immutable tuples. Their suffixes
    may differ. Actual text encoding and surface checks belong to the caller.

    >>> bare, spaced = [10, 11, 12], [31]
    >>> paths = CandidatePaths(bare, spaced)
    >>> bare.append(13)
    >>> spaced[0] = 32
    >>> paths.bare, paths.spaced
    ((10, 11, 12), (31,))
    """

    bare: TokenPath
    spaced: TokenPath

    def __init__(self, bare: Sequence[int], spaced: Sequence[int]) -> None:
        object.__setattr__(self, "bare", _token_path(bare))
        object.__setattr__(self, "spaced", _token_path(spaced))


@dataclass(frozen=True)
class ReadoutRequest:
    """Required token logits after a prefix of a selected continuation.

    The caller appends prefix to the unchanged context before requesting the
    distinct wanted ids. An empty prefix means the selected root; its scores
    are reused from the initial form-selection readout.

    >>> context = (7, 8)
    >>> request = ReadoutRequest(prefix=(10,), wanted=(11, 99, 100))
    >>> context + request.prefix, request.wanted
    ((7, 8, 10), (11, 99, 100))
    >>> context
    (7, 8)
    """

    prefix: TokenPath
    wanted: tuple[int, ...]


@dataclass(frozen=True)
class TriePlan:
    """A selected trie projected onto only its branching readouts.

    Build with build_trie. Paths preserve candidate order and are immutable;
    readouts use deterministic parent-first prefix order. Forced transitions
    and leaves need no readout, but their tokens remain in complete paths and
    in prefixes of later requests. The plan contains no context or scores.

    >>> plan = build_trie({"up": (10,), "down": (20,)})
    >>> plan.paths
    (('up', (10,)), ('down', (20,)))
    >>> plan.readouts
    (ReadoutRequest(prefix=(), wanted=(10, 20)),)
    >>> logs = plan.score({(): {10: 2.0, 20: 2.0}})
    >>> [round(math.exp(value), 3) for value in logs.values()]
    [0.5, 0.5]
    """

    paths: tuple[tuple[str, TokenPath], ...]
    end_tokens: tuple[int, ...]
    readouts: tuple[ReadoutRequest, ...]

    @property
    def additional_readouts(self) -> tuple[ReadoutRequest, ...]:
        """Non-root branches needing scores after the initial selection readout.

        >>> plan = build_trie(
        ...     {"up": (10,), "upward": (10, 11, 12), "down": (20,)},
        ...     end_tokens=(99, 100),
        ... )
        >>> [request.prefix for request in plan.readouts]
        [(), (10,)]
        >>> plan.additional_readouts
        (ReadoutRequest(prefix=(10,), wanted=(11, 99, 100)),)
        """
        return tuple(request for request in self.readouts if request.prefix)

    def score(self, readouts: Mapping[TokenPath, Mapping[int, float]]) -> dict[str, float]:
        """Return natural-log credences from all required branch readouts.

        Keys are continuation prefixes, not full contexts. Reuse the initial
        form scores under () when the selected root branches. Every requested
        branch and token must be present and valid, even at zero incoming mass.
        Extra contexts or tokens are ignored. Forced steps contribute zero;
        a terminal with children adds the grouped END event. No label-level
        renormalization, model calls, or cross-call state is introduced.

        Illustrative ids only: the root splits equally between 10 and 20.
        At (10,), two END ids give stopping mass 2/3 and continuation mass
        1/3. The final token 12 is forced and needs no score.

        >>> forms = {
        ...     "up": CandidatePaths((10,), (30,)),
        ...     "upward": CandidatePaths((10, 11, 12), (31,)),
        ...     "down": CandidatePaths((20,), (40,)),
        ... }
        >>> validate_potential_paths(forms, end_tokens=(99, 100))
        >>> initial = {10: 2.0, 20: 2.0, 30: 1.0, 31: 0.0, 40: 1.0}
        >>> selected = select_paths(forms, initial)
        >>> plan = build_trie(selected, end_tokens=(99, 100))
        >>> logs = plan.score({(): initial, (10,): {11: 0.0, 99: 0.0, 100: 0.0}})
        >>> {label: round(math.exp(value), 6) for label, value in logs.items()}
        {'up': 0.333333, 'upward': 0.166667, 'down': 0.5}
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
    """Distinct first ids of both complete forms, in deterministic sorted order.

    Validate candidate/path ownership without reading scores. Stopping
    boundaries additionally require validate_potential_paths before inference.

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
    """Select one entire path per candidate by its first-token logit.

    Choose spaced only when its score is strictly larger; exact finite or
    -inf ties choose the named bare form. Validate every first score of both
    forms, including discarded ones. Preserve candidate order and the chosen
    complete suffix. Selection contributes no probability factor.

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
    """Preflight both forms before any score readout.

    Reject cross-label path ownership collisions. A possible strict prefix
    overlap across labels requires a nonempty END set disjoint from the next
    continuation token. Relationships between one candidate's own forms do
    not require END: those paths cannot coexist in the selected trie.

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
    """Compile one selected complete path per candidate into a pure plan.

    Merge shared prefixes before defining allowed-token sets. Reject empty or
    cross-label identical paths. At a terminal with children, require usable
    END ids and include each distinct id in that node's normalizer. Leaves
    stop without end scores; singleton edges are forced. Copy all input data
    into immutable plan fields, without caching or predicting future selection.

    >>> plan = build_trie(
    ...     {"up": (10,), "upward": (10, 11, 12), "down": (20,)},
    ...     end_tokens=(99, 100),
    ... )
    >>> [(request.prefix, request.wanted) for request in plan.readouts]
    [((), (10, 20)), ((10,), (11, 99, 100))]
    >>> (10, 11) not in {request.prefix for request in plan.readouts}
    True
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
    """Allow repeated forms for one label, but reject cross-label ownership.

    >>> _path_owners((("up", (10,)), ("up", (10,))))
    {(10,): 'up'}
    >>> _path_owners((("up", (10,)), ("down", (10,))))
    Traceback (most recent call last):
        ...
    ValueError: Path (10,) is owned by both 'up' and 'down'
    """
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
    """Canonicalize END ids so repeated ids cannot multiply stopping mass.

    >>> _end_tokens([100, 99, 100])
    (99, 100)
    """
    return tuple(sorted(set(_token_ids(values))))


def _validate_stopping_boundary(
    prefix: TokenPath, continuations: Iterable[int], end_tokens: tuple[int, ...]
) -> None:
    """Keep stopping ids disjoint from the same node's continuation ids.

    >>> _validate_stopping_boundary((10,), (11,), (99, 100))
    >>> _validate_stopping_boundary((10,), (11,), (11, 99))
    Traceback (most recent call last):
        ...
    ValueError: END overlaps continuation tokens [11] at prefix (10,)
    """
    if not end_tokens:
        raise ValueError(f"Prefix {prefix} needs a nonempty END set")
    overlap = set(end_tokens).intersection(continuations)
    if overlap:
        raise ValueError(f"END overlaps continuation tokens {sorted(overlap)} at prefix {prefix}")
