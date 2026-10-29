"""Measure a language model's probabilities over a fixed set of answers.

Running a model is not supported yet. This version provides the result types
and the scoring math: given token IDs for each answer and a model's logits,
credences.trie scores both spellings of each answer and
credences.probabilities.measurement_from_raw turns the result into a Measurement.
See docs/pure-core.md in the repository for a walkthrough.
"""

from .probabilities import (
    Measurement,
    RawReadout,
    ScoredPath,
    choose_top_label,
    top_label_weights,
)

__all__ = ["Measurement", "RawReadout", "ScoredPath", "choose_top_label", "top_label_weights"]
