"""Measure a language model's probabilities over a fixed set of answers.

Running a model is not supported yet. This version provides the result types
and the scoring math: given token IDs for each answer and a model's logits,
credences.trie computes each answer's probability and
credences.probabilities.measurement_from_raw turns the result into a Measurement.
See docs/pure-core.md in the repository for a walkthrough, and
docs/tokenization.md for how answers are matched to a model's tokens.
"""

from .probabilities import (
    Measurement,
    RawReadout,
    ScoredPath,
    choose_top_label,
    top_label_weights,
)

__all__ = ["Measurement", "RawReadout", "ScoredPath", "choose_top_label", "top_label_weights"]
