# Changelog

## Unreleased

- Add named complete-path selection, conservative alternative-path preflight,
  and immutable selected-trie readout plans. Shared prefixes normalize once;
  leaves and singleton transitions require no additional scores.
- Wire production selection and trie scoring into the independent Decimal
  parity property across generated paths and golden examples. Add recorded
  fake-readout tests for root reuse, stopping events, and exact token contexts.
- Add pure masked log probabilities, complete requested-score validation, log
  event aggregation, confidence summaries, and caller-controlled tie helpers.
- Clarify SPEC §5.5: backend scores are cast to float32, while pure calculations
  use Python's higher-precision floats. Strict float32 subtraction can overflow
  between two finite float32 extremes; carrying the calculation in higher
  precision preserves finite logs and keeps the core dependency-free. The
  specified selection, masking, forcing, and fallback rules are unchanged.
- Document the result builder's normalization check (`1e-12` relative tolerance,
  no label-level renormalization) and boundary-roundoff bounds for confidence.
- Bound tiny positive grouped-event roundoff at log probability 0, with excess
  event mass beyond the same tolerance remaining an error. General logsumexp
  stays unbounded; raw results are not repaired or renormalized.
- Construct the exact-log winner regression's rounded-probability tie using
  the current math library rather than assuming identical exp rounding on
  macOS and Linux.
