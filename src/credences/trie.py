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
    """

    paths: tuple[tuple[str, TokenPath], ...]
    end_tokens: tuple[int, ...]
    readouts: tuple[ReadoutRequest, ...]

    @property
    def additional_readouts(self) -> tuple[ReadoutRequest, ...]:
        """Non-root branches needing scores after the initial selection readout."""
        return tuple(request for request in self.readouts if request.prefix)

    def score(self, readouts: Mapping[TokenPath, Mapping[int, float]]) -> dict[str, float]:
        """Return natural-log credences from all required branch readouts.

        Keys are continuation prefixes, not full contexts. Reuse the initial
        form scores under () when the selected root branches. Every requested
        branch and token must be present and valid, even at zero incoming mass.
        Extra contexts or tokens are ignored. Forced steps contribute zero;
        a terminal with children adds the grouped END event. No label-level
        renormalization, model calls, or cross-call state is introduced.
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
    return tuple(sorted(set(_token_ids(values))))


def _validate_stopping_boundary(
    prefix: TokenPath, continuations: Iterable[int], end_tokens: tuple[int, ...]
) -> None:
    if not end_tokens:
        raise ValueError(f"Prefix {prefix} needs a nonempty END set")
    overlap = set(end_tokens).intersection(continuations)
    if overlap:
        raise ValueError(f"END overlaps continuation tokens {sorted(overlap)} at prefix {prefix}")
