"""Compute forced-choice probabilities and interpret candidate distributions.

Token-score helpers accept raw model logits and return natural-log probabilities.
Result types expose the candidate distribution, confidence summaries, and the
numerical detail needed to inspect or save a result. No function queries a model.
Raw scores should be float32-range logits supplied as Python floats, or -inf.
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
    """The numerical detail behind a candidate probability distribution.

    Use this record to inspect very small probabilities and which answer
    representations were scored. ``credence_logprobs`` contains the natural log
    of each candidate's probability, not raw token logits. A finite log remains
    informative even when its ordinary probability underflows to zero; ``-inf``
    means exact zero mass.

    ``scored_token_ids`` maps each candidate answer string to the token IDs of its
    selected complete continuation, which may be the bare or leading-space form.
    IDs identify entries in the model's vocabulary, not individual characters.
    ``context_token_count`` is the length of the fixed prompt context. Paths
    exclude that context; discarded alternatives and branch scores are not stored.

    Suppose an illustrative tokenizer encodes ``"very positive"`` as two pieces:
    ``"very"`` (ID 10) and ``" positive"`` (ID 11), while ``"negative"`` is one
    token (ID 20). If these bare forms are selected, the record contains:

    >>> raw = RawReadout(
    ...     credence_logprobs={"very positive": math.log(0.75), "negative": math.log(0.25)},
    ...     scored_token_ids={"very positive": (10, 11), "negative": (20,)},
    ...     context_token_count=7,
    ... )
    >>> raw.scored_token_ids["very positive"]
    (10, 11)
    """

    credence_logprobs: dict[str, float]
    scored_token_ids: dict[str, tuple[int, ...]]
    context_token_count: int


@dataclass(frozen=True)
class Measurement:
    """A candidate probability distribution and summaries for interpreting it.

    Read ``credences`` when you need the full forced-choice distribution, rather
    than only a hard label. These probabilities describe the supplied candidates
    under the prompt and representation constraints, not the full vocabulary or
    a guarantee of calibration. ``raw`` provides log credences and selected paths.

    ``entropy`` measures uncertainty across the distribution in bits.
    ``top_labels`` contains every exact maximizer of log credence in candidate
    order, without rounding or random tie-breaking.

    The three confidence summaries answer different questions:

    - ``top_label_confidence``: how strongly is the best label favored over
      uniform guessing? For K candidates, it is (p1 - 1/K) / (1 - 1/K).
    - ``entropy_confidence``: how concentrated is the whole distribution?
      It is 1 - entropy / log2(K).
    - ``margin_confidence``: how far ahead is the best label of its nearest
      competitor? It is p1 - p2, with p1 >= p2 the two largest probabilities.

    Fields cannot be reassigned, but the dictionaries are not deeply immutable.
    """

    credences: dict[str, float]
    entropy: float
    top_labels: tuple[str, ...]
    top_label_confidence: float
    entropy_confidence: float
    margin_confidence: float
    raw: RawReadout

    def to_dict(self) -> dict[str, object]:
        """Export a measurement for storage or sharing as strict JSON.

        Return a dictionary with the same named fields and a nested ``raw`` record.
        Winner tuples and token paths become lists. Only negative-infinity
        log credences become ``None`` (JSON null); finite logs are preserved,
        including when the corresponding ordinary credence is zero.

        Editing the returned containers does not change the measurement.
        Invalid nonfinite values raise ValueError rather than producing
        nonstandard JSON. No measurement values are rounded or recalculated.

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
    """Require complete, valid token scores before using a model readout.

    ``wanted`` identifies the token IDs you requested; ``readout`` supplies their
    raw logits. Return a dictionary containing only those IDs. Missing scores,
    NaN, or positive infinity raise BackendReadoutError: they cannot be treated
    as zero weights. Finite logits, including positive values, and ``-inf`` are
    valid. Unrequested IDs are ignored.

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
    """Find next-token probabilities when only the allowed tokens may be chosen.

    Supply raw logits in ``readout`` and the permitted token IDs in ``allowed``.
    Return an ID-to-natural-log-probability dictionary, normalized over distinct
    allowed IDs only, not the full vocabulary.

    One allowed token is forced with probability 1 and needs no score. If all
    allowed scores are ``-inf``, use a uniform token distribution; otherwise
    individual ``-inf`` scores get zero mass. Missing or malformed branch scores
    raise BackendReadoutError. An empty allowed set raises ValueError.

    When several tokens mean stopping, include each END ID alongside continuing
    tokens, then combine their returned log probabilities with logprob_sum.

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
    """Find a summed weight while keeping inputs and output in log space.

    Supply natural-log weights and receive the natural log of their summed
    weight. This does not normalize the inputs: a positive result is valid
    when the total weight exceeds 1. For a normalized probability event, use
    logprob_sum instead. Empty or all-``-inf`` input means zero total weight and
    returns ``-inf``; NaN or positive infinity raises ValueError.

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
    """Find the probability of an event with several mutually exclusive outcomes.

    Supply normalized natural-log probabilities, not raw logits or ordinary
    probabilities. Return the natural log of their summed probability. For
    example, multiple END tokens can all represent choosing to stop.

    Input logs must be nonpositive, and event mass must not exceed 1 beyond
    relative tolerance 1e-12. Violations raise ValueError. Tiny positive output
    caused by roundoff is bounded at 0; negative logs are preserved. Empty or
    all-zero-mass events return ``-inf``.

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
    """Turn scored candidate answers into a distribution and confidence summaries.

    Supply a RawReadout containing normalized candidate log probabilities and
    one selected token path per candidate. These are final candidate log
    credences, not raw token logits or unnormalized answer likelihoods. Return
    a Measurement with probabilities, all exact winners, entropy in bits, and
    the three confidence summaries, preserving the input's candidate order.

    At least two candidates and matching path keys are required. Implied
    probability mass must sum to 1 within relative tolerance 1e-12; invalid
    distributions raise ValueError rather than being renormalized. Raw logs
    stay unchanged, including finite logs whose probabilities underflow.
    Confidence summaries lie in [0, 1], allowing for boundary roundoff.

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
    """Choose one hard label when an application must resolve exact winners.

    Measurement preserves every winner in ``top_labels``. Call this helper
    explicitly to select one uniformly, using your own random.Random-compatible
    generator. The distribution and winner set do not change; the supplied
    generator owns the random state.

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
    """Count winning labels without arbitrarily breaking ties.

    Return a dictionary assigning each of N exact winners weight 1/N, omitting
    non-winners. Add these dictionaries across measurements for fractional
    winner counts. These weights are not the candidate probabilities.
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
    """Return a finite maximum and the log normalizer relative to it."""
    # Subtracting the maximum keeps exponentials <= 1, preventing overflow.
    # Keep the correction separate: adding it to a large maximum can erase it.
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
