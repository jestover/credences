"""Backend-free result containers and JSON serialization (SPEC §6).

Probability calculations will be added separately. These dataclasses hold
already-computed values; constructing them does not perform measurement.
"""

import math
from dataclasses import dataclass


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


def _finite(value: float, field: str) -> float:
    if not math.isfinite(value):
        raise ValueError(
            f"{field} must contain finite values for JSON serialization; got {value!r}"
        )
    return value
