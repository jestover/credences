"""Pure probability rules, confidence summaries, and result persistence.

Backend scores arrive as float32 values. Computation uses Python floats with
stable, max-shifted log arithmetic; no tokenizer or runtime is imported.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Protocol

from .errors import BackendReadoutError

# Shared tolerance for normalized event mass and complete result mass.
_MASS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class RawReadout:
    """Log credences, selected complete token paths, and fixed-context length.

    Log credences use natural logarithms. Negative infinity means exact zero
    mass; a finite log remains meaningful even when its probability underflows.
    Paths contain only the selected form, not discarded forms or a full trace.

    A multi-token selected answer keeps its whole path, excluding context:

    >>> raw = RawReadout(
    ...     credence_logprobs={"A": math.log(0.75), "B": math.log(0.25)},
    ...     scored_token_ids={"A": (10, 11), "B": (20,)},
    ...     context_token_count=7,
    ... )
    >>> raw.scored_token_ids["A"]
    (10, 11)
    """

    credence_logprobs: dict[str, float]
    scored_token_ids: dict[str, tuple[int, ...]]
    context_token_count: int


@dataclass(frozen=True)
class Measurement:
    """The credence vector, its derived summaries, and the raw readout.

    ``credences`` maps each candidate to its forced-choice probability.
    ``entropy`` is Shannon entropy in bits, with zero-mass terms omitted.
    ``top_labels`` contains every exact maximizer of log credence in candidate
    order, without rounding or random tie-breaking.

    For K candidates and sorted probabilities p1 >= p2 >= ...:
    ``top_label_confidence`` is (p1 - 1/K) / (1 - 1/K);
    ``entropy_confidence`` is 1 - entropy / log2(K);
    ``margin_confidence`` is p1 - p2.

    Fields cannot be reassigned, but the dictionaries are not deeply immutable.
    No run metadata or reasoning policy is stored on the result.
    """

    credences: dict[str, float]
    entropy: float
    top_labels: tuple[str, ...]
    top_label_confidence: float
    entropy_confidence: float
    margin_confidence: float
    raw: RawReadout

    def to_dict(self) -> dict[str, object]:
        """Return a detached, JSON-safe snapshot with the same named fields.

        ``raw`` is a nested dictionary with the three RawReadout field names.
        Winner tuples and token paths become lists. Only negative-infinity
        log credences become ``None`` (JSON null); finite logs are preserved,
        including when the corresponding ordinary credence is zero.

        Nonfinite credences or summaries, and NaN/positive-infinity log
        credences, raise ValueError rather than leaking invalid JSON values.
        This method does not recompute, round, or normalize any measurement.

        Underflowed finite logs survive; exact-zero logs become JSON null:

        >>> import json
        >>> raw = RawReadout(
        ...     {"A": 0.0, "tiny": -1000.0, "zero": -math.inf},
        ...     {"A": (1,), "tiny": (2,), "zero": (3,)}, 7,
        ... )
        >>> result = measurement_from_raw(raw)
        >>> exported = result.to_dict()
        >>> exported["raw"]["credence_logprobs"]
        {'A': 0.0, 'tiny': -1000.0, 'zero': None}
        >>> exported["raw"]["scored_token_ids"]["A"]
        [1]
        >>> json.loads(json.dumps(exported, allow_nan=False))["top_labels"]
        ['A']
        """
        return {
            "credences": {
                label: _finite(value, "credences") for label, value in self.credences.items()
            },
            "entropy": _finite(self.entropy, "entropy"),
            "top_labels": list(self.top_labels),
            "top_label_confidence": _finite(self.top_label_confidence, "top_label_confidence"),
            "entropy_confidence": _finite(self.entropy_confidence, "entropy_confidence"),
            "margin_confidence": _finite(self.margin_confidence, "margin_confidence"),
            "raw": {
                "credence_logprobs": {
                    label: None if value == -math.inf else _finite(value, "credence_logprobs")
                    for label, value in self.raw.credence_logprobs.items()
                },
                "scored_token_ids": {
                    label: list(path) for label, path in self.raw.scored_token_ids.items()
                },
                "context_token_count": self.raw.context_token_count,
            },
        }


def validate_readout(wanted: Iterable[int], readout: Mapping[int, float]) -> dict[int, float]:
    """Copy and validate every distinct requested score, ignoring other ids.

    Use this for the complete initial selection readout, even if its selected
    root later becomes forced. Missing/NaN/+inf scores are contract violations,
    never zeros. Negative infinity is valid and remains negative infinity.

    Positive logits are valid; repeated wanted IDs are checked once:

    >>> validate_readout([10, 10, 20], {10: 2.0, 20: -math.inf, 99: 8.0})
    {10: 2.0, 20: -inf}
    """
    scores = {}
    for token in dict.fromkeys(wanted):
        try:
            value = readout[token]
        except KeyError as error:
            raise BackendReadoutError(f"Missing requested score for token {token}") from error
        try:
            scores[token] = _log_weight(value, f"score for token {token}")
        except (TypeError, ValueError) as error:
            raise BackendReadoutError(str(error)) from error
    return scores


def masked_logprobs(allowed: Iterable[int], readout: Mapping[int, float]) -> dict[int, float]:
    """Return natural-log probabilities over distinct allowed token ids.

    A singleton is forced without inspecting the readout. Branches validate
    every allowed score, then normalize in log space. All -inf scores yield a
    token-uniform distribution; otherwise individual -inf scores retain zero
    mass. END ids are ordinary tokens here; callers aggregate their logprobs
    with logprob_sum rather than treating END as one token.

    Normalize logits, not already-normalized probabilities. Round only display:

    >>> logs = masked_logprobs([10, 20], {10: 2.0, 20: 1.0})
    >>> {token: round(math.exp(value), 6) for token, value in logs.items()}
    {10: 0.731059, 20: 0.268941}
    >>> masked_logprobs([12], {})  # Forced: no score required.
    {12: 0.0}
    >>> fallback = masked_logprobs([10, 20], {10: -math.inf, 20: -math.inf})
    >>> [round(math.exp(value), 6) for value in fallback.values()]
    [0.5, 0.5]
    """
    tokens = tuple(dict.fromkeys(allowed))
    if not tokens:
        raise ValueError("The allowed token set must be nonempty")
    if len(tokens) == 1:
        return {tokens[0]: 0.0}
    scores = validate_readout(tokens, readout)
    values = tuple(scores.values())
    if all(value == -math.inf for value in values):
        return dict.fromkeys(tokens, -math.log(len(tokens)))
    maximum, correction = _shifted_normalizer(values)
    # Subtract the maximum first. score - (maximum + correction) would lose
    # the correction at large common offsets, even for perfectly equal scores.
    return {token: (score - maximum) - correction for token, score in scores.items()}


def logsumexp(values: Iterable[float]) -> float:
    """Natural log of summed weights, including empty/all-zero events.

    Empty input or all -inf returns -inf. NaN and +inf raise ValueError.
    Positive output is valid for arbitrary weights. Use logprob_sum for
    grouped normalized events such as END; never sum alternative forms.

    Two unit weights sum to 2, so their summed log weight is positive:

    >>> round(logsumexp([0.0, 0.0]), 6)
    0.693147
    >>> logsumexp([])
    -inf
    """
    weights = tuple(_log_weight(value, "log weight") for value in values)
    if not weights or all(value == -math.inf for value in weights):
        return -math.inf
    maximum, correction = _shifted_normalizer(weights)
    return maximum + correction


def logprob_sum(values: Iterable[float]) -> float:
    """Sum a normalized event in log space, with roundoff bounded at log(1).

    Each input must be a nonpositive log probability. The sum may exceed 1
    only within the shared normalization tolerance; otherwise raise.
    Tiny positive output caused by grouping roundoff becomes 0. Negative
    logs, including tiny finite values and -inf, are never rounded away.

    Two END tokens compete with one continuation token. Group only after
    normalizing all three distinct IDs:

    >>> steps = masked_logprobs([11, 99, 100], {11: 0.0, 99: 0.0, 100: 0.0})
    >>> stopping = logprob_sum(steps[token] for token in (99, 100))
    >>> round(math.exp(stopping), 6)
    0.666667
    """
    probabilities = tuple(_log_weight(value, "log probability") for value in values)
    if any(value > 0 for value in probabilities):
        raise ValueError("A log probability cannot be positive")
    result = logsumexp(probabilities)
    if result > math.log1p(_MASS_TOLERANCE):
        raise ValueError("Grouped event probability exceeds 1")
    return min(0.0, result)


def measurement_from_raw(raw: RawReadout) -> Measurement:
    """Internal result builder: derive summaries from normalized log credences.

    Raw mapping order defines candidate order. At least two candidates and
    matching path keys are required. Mass must sum to 1 within a relative
    tolerance of 1e-12; malformed distributions raise instead of receiving a
    label-level renormalization or fallback. Raw logs are never rewritten.

    Confidence summaries are bounded to [0, 1] to remove boundary roundoff;
    credences and entropy are not rounded or renormalized. Input dictionaries
    are copied so subsequent caller edits cannot change the built result.

    >>> raw = RawReadout(
    ...     {"A": math.log(0.6), "B": math.log(0.2), "C": math.log(0.2)},
    ...     {"A": (10,), "B": (20,), "C": (30,)}, 7,
    ... )
    >>> result = measurement_from_raw(raw)
    >>> result.top_labels
    ('A',)
    >>> tuple(round(value, 6) for value in (
    ...     result.top_label_confidence, result.entropy_confidence, result.margin_confidence,
    ... ))
    (0.4, 0.135026, 0.4)
    >>> raw.credence_logprobs["A"] = -math.inf  # Input edits do not change the snapshot.
    >>> result.raw.credence_logprobs["A"] == math.log(0.6)
    True
    """
    if len(raw.credence_logprobs) < 2:
        raise ValueError("A measurement requires at least two candidates")
    if raw.credence_logprobs.keys() != raw.scored_token_ids.keys():
        raise ValueError("Log credences and scored paths must have the same candidates")
    logs = {
        label: _log_weight(value, f"log credence for {label!r}")
        for label, value in raw.credence_logprobs.items()
    }
    if any(value > 0 for value in logs.values()):
        raise ValueError("A log credence cannot be positive")
    credences = {label: math.exp(value) for label, value in logs.items()}
    total = math.fsum(credences.values())
    if not math.isclose(total, 1.0, rel_tol=_MASS_TOLERANCE, abs_tol=0.0):
        raise ValueError(f"Credences must sum to 1; got {total!r}")
    maximum = max(logs.values())
    winners = tuple(label for label, value in logs.items() if value == maximum)
    entropy = -math.fsum(
        probability * logs[label] for label, probability in credences.items() if probability != 0.0
    ) / math.log(2)
    largest, second = sorted(credences.values(), reverse=True)[:2]
    count = len(credences)
    # frozen=True prevents field reassignment, not edits to dictionary contents.
    # Own the mappings so changes to the input cannot invalidate these summaries.
    snapshot = RawReadout(
        credence_logprobs=logs,
        scored_token_ids={label: tuple(raw.scored_token_ids[label]) for label in logs},
        context_token_count=raw.context_token_count,
    )
    return Measurement(
        credences=credences,
        entropy=entropy,
        top_labels=winners,
        top_label_confidence=_bounded_confidence((largest - 1 / count) / (1 - 1 / count)),
        entropy_confidence=_bounded_confidence(1 - entropy / math.log2(count)),
        margin_confidence=largest - second,
        raw=snapshot,
    )


class _ChoiceRng(Protocol):
    """Typing interface for a caller-owned RNG; no runtime validation or state."""

    def choice(self, seq: Sequence[str], /) -> str: ...


def choose_top_label(measurement: Measurement, *, rng: _ChoiceRng) -> str:
    """Uniformly choose an exact winner using only the caller's RNG.

    >>> import random
    >>> tied = measurement_from_raw(RawReadout(
    ...     {"A": -math.log(2), "B": -math.log(2)},
    ...     {"A": (10,), "B": (20,)}, 7,
    ... ))
    >>> choose_top_label(tied, rng=random.Random(7)) in tied.top_labels
    True
    """
    if not measurement.top_labels:
        raise ValueError("A measurement must have at least one top label")
    return rng.choice(measurement.top_labels)


def top_label_weights(measurement: Measurement) -> dict[str, float]:
    """Return equal counting weights for exact winners, not their credences.

    With N winners, each gets weight 1/N in top_labels order; non-winners
    are omitted. Summing across measurements gives fractional winner counts.
    For example, two winners with credence 0.4 each receive counting weight
    0.5 each. The measurement is unchanged. An empty winner set raises.

    >>> result = measurement_from_raw(RawReadout(
    ...     {"A": math.log(0.4), "B": math.log(0.4), "C": math.log(0.2)},
    ...     {"A": (10,), "B": (20,), "C": (30,)}, 7,
    ... ))
    >>> top_label_weights(result)
    {'A': 0.5, 'B': 0.5}
    """
    if not measurement.top_labels:
        raise ValueError("A measurement must have at least one top label")
    return dict.fromkeys(measurement.top_labels, 1 / len(measurement.top_labels))


def _shifted_normalizer(values: Sequence[float]) -> tuple[float, float]:
    """Return (m, correction), where m is the maximum input log weight.

    correction = log(sum(exp(value - m))) over all input values.
    Subtracting m makes each exponential at most 1, avoiding overflow.
    Keep m and correction separate: adding them first can lose the small
    correction at large offsets, corrupting subsequent log probabilities.

    One maximum contributes exactly 1. Exclude it from the tail and use
    log1p(tail) to retain small corrections that log(1 + tail) would lose to
    rounding. fsum reduces rounding when adding the remaining weights.
    Input must be nonempty and contain at least one finite value.

    Large logits remain safe, and tiny tail corrections are not rounded away:

    >>> maximum, correction = _shifted_normalizer((1000.0, 999.0))
    >>> maximum, round(correction, 6)
    (1000.0, 0.313262)
    >>> _, tiny_correction = _shifted_normalizer((0.0, -40.0))
    >>> tiny_correction > 0.0
    True
    """
    maximum = max(values)
    peak = values.index(maximum)
    # Exclude one maximum (weight 1), then use log1p to preserve corrections
    # smaller than machine epsilon instead of rounding 1 + tiny mass to 1.
    tail = math.fsum(
        math.exp(value - maximum) for index, value in enumerate(values) if index != peak
    )
    return maximum, math.log1p(tail)


def _log_weight(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number; got {value!r}")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} is outside the supported numeric range") from error
    if math.isnan(number) or number == math.inf:
        raise ValueError(f"{field} must be finite or -inf; got {number!r}")
    return number


def _bounded_confidence(value: float) -> float:
    return min(1.0, max(0.0, value))


def _finite(value: float, field: str) -> float:
    if not math.isfinite(value):
        raise ValueError(
            f"{field} must contain finite values for JSON serialization; got {value!r}"
        )
    return value
