"""Measure a language model's probabilities over a fixed set of answers.

Running a model is not supported yet. This version provides the result types
and the scoring math: given token IDs for each answer and a model's logits,
credences.trie computes each answer's credence and
credences.probabilities.measurement_from_raw turns those into a Measurement.
See docs/pure-core.md in the repository for a walkthrough.
"""

from .probabilities import Measurement, RawReadout, choose_top_label, top_label_weights

__all__ = ["Measurement", "RawReadout", "choose_top_label", "top_label_weights"]
