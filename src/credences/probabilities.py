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
    """

    credence_logprobs: dict[str, float]
    scored_token_ids: dict[str, tuple[int, ...]]
    context_token_count: int


@dataclass(frozen=True)
class Measurement:
    """The credence vector, its derived summaries, and the raw readout.

    Entropy is in bits. ``top_labels`` contains all exact winners in candidate
    order. Fields cannot be reassigned, but the dictionaries are not deeply
    immutable. No run metadata or reasoning policy is stored on the result.
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
    """Apply SPEC §5.1's transition rule over distinct allowed token ids.

    A singleton is forced without inspecting the readout. Branches validate
    every allowed score, then normalize in log space. All -inf scores yield a
    token-uniform distribution; otherwise individual -inf scores retain zero
    mass. END ids are ordinary tokens here; callers aggregate their logprobs
    with logprob_sum rather than treating END as one token.
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
    def choice(self, seq: Sequence[str], /) -> str: ...


def choose_top_label(measurement: Measurement, *, rng: _ChoiceRng) -> str:
    """Uniformly choose an exact winner using only the caller's RNG."""
    if not measurement.top_labels:
        raise ValueError("A measurement must have at least one top label")
    return rng.choice(measurement.top_labels)


def top_label_weights(measurement: Measurement) -> dict[str, float]:
    """Fractional winner-count weights, not candidate credences (SPEC §6.4)."""
    if not measurement.top_labels:
        raise ValueError("A measurement must have at least one top label")
    return dict.fromkeys(measurement.top_labels, 1 / len(measurement.top_labels))


def _shifted_normalizer(values: Sequence[float]) -> tuple[float, float]:
    """Return the finite maximum and log normalizer relative to it."""
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
