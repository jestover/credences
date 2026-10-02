# Changelog

## Unreleased

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
