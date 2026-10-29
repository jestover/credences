# Changelog

## Unreleased

- Revise the measurement design (SPEC §4–§8, §10, §14; DECISIONS D3, D5, D14,
  D16; AGENTS): candidates are trimmed at the ends; each candidate's credence is
  the sum of its bare and single-space spellings, scored in one trie, replacing
  first-token selection of one spelling; spellings are encoded together with the
  prompt and scored from the longest token prefix they share with it, with exact
  round trips; prefills are kept exactly, restoring trailing whitespace that
  chat templates trim; and `RawReadout` gains per-path probabilities and a
  `coverage` diagnostic, which needs the backend to return a full-vocabulary log
  normalizer. Based on tests of nine tokenizer families and Geh et al. (EMNLP
  2024). The pure core still implements selection; PLAN Phase 1b brings the code
  in line.
- Complete Phase 1: add chat message types with structural validation
  (`credences.types`), the `Tokenizer` and `FullLogitsBackend` interfaces, and a
  protocol-conforming fake tokenizer and backend for tests. ty checks the fakes
  against the interfaces, and import-linter now keeps `credences.tokenization`
  free of model runtimes.
- Clarify SPEC §10.2: the `Tokenizer` protocol declares no default arguments.
  Hugging Face tokenizers default to adding special tokens, so relying on
  defaults could silently put a beginning-of-text token into an answer path;
  callers pass every flag explicitly instead.
- Rewrite the README, docstrings, and pure-core walkthrough for readers new to
  the project: what the package does and why, plain-language definitions,
  and fewer internal terms. Move checkout-synchronization notes from the README
  to `AGENTS.md`, run the README example as a doctest, and add newcomer
  guidance to `docs/style.md`.
- Adopt Google-style API docstrings and explicit Ruff formatting/lint conventions.
  Add ty as a locked development dependency and check library types in ordinary CI.
- Refocus API docstrings on purpose, context, input/output meaning, and usage.
  Explain whitespace-dependent candidate tokenization, and keep non-obvious
  implementation rationale in concise code comments instead.
- Add small runnable examples to the core docstrings and a durable pure-core
  walkthrough covering selection, trie scoring, results, JSON, and ties.
  Run the examples as doctests in ordinary CI to prevent documentation drift.
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
