# AGENTS.md — working agreement for `credences`

Vendor-neutral guidance for *any* coding agent (Claude, Cursor, Copilot, Aider, …)
or human working in this repo. `CLAUDE.md` and `.cursorrules` should be one-line
stubs pointing here.

> **Read `SPEC.md` §1–§8 first** — it defines every term used below (Credoscope,
> masked semantics, trie, blessed registry). Then `DECISIONS.md` for why.
> New to the project? Do not skip that order; this file will read as jargon
> otherwise.

## What this project is

`credences` extracts a model's **credence vector** — its probability distribution
over a small set of candidate answers — directly from next-token probabilities.
One core object: a **`Credoscope`** built once by
`build_credoscope(model, candidates, …)`, then `scope.measure(...) -> Measurement` per
passage. A `Measurement` returns the vector first, plus derived summaries such as
`top_label_confidence`, `entropy_confidence`, and `margin_confidence`. It
requires at least two candidates. `.measure()` accepts one input and returns a
`Measurement`, or accepts a sequence and returns one aligned `Measurement` per
input.

`credences` is **generic measurement machinery only**. Do not add domain logic
(transcripts, earnings calls, portfolios, event studies) — that lives in the
consuming analysis repo.

## The two rules that govern structure

**1. Keep the dependency tiers intact** (enforced by import-linter in CI):

- *Pure* (no tokenizer, no backend imports): `trie.py`, `probabilities.py`,
  `types.py`, `errors.py`.
- *Tokenizer-bound, backend-free* (HF tokenizer objects OK; no runtime imports):
  `tokenization.py`, `reasoning.py`, `model.py`.
- *Backend-bound*: `backends/*`, and `credoscope.py` — the **single chokepoint**
  that drives a backend.

**2. The backend protocol stays thin and honest.** One capability contract:
arbitrary-token logits at supplied contexts (`FullLogitsBackend`, SPEC §10).
V1 implements MLX only. Never add low-level KV-cache operations,
generic `generate()`, or scheduling to a protocol; caching is backend-internal.
V1 has no generation capability: `reasoning="bypass"` is its only supported mode
(SPEC §9). A backend that cannot do something honestly must raise, never
approximate silently.

## Measurement correctness is the product

- Credences are the **forced-choice (masked) distribution** — requested-token
  logits from the backend, renormalized over the allowed set at each branch node
  by the Credoscope (SPEC §5). Any
  change to probability math updates the trie-vs-naive **parity property test
  first** (SPEC §14). Compare both first-token scores, select one complete path
  per candidate, and score only the selected trie. Merge shared prefixes before
  normalization; never count a shared first token once per label.
- Force singleton transitions to probability 1 without querying them. All-`-inf`
  branches use uniform probability over distinct allowed token ids. With any
  finite score, `-inf` tokens get zero. Missing/NaN/positive-infinity scores remain
  errors; fallback is not a repair for malformed backend output.
- The answer boundary (SPEC §8) and reasoning inserts (SPEC §9.1) are the two
  easiest places to silently corrupt every number. Touch them only with Tier-2
  tokenizer tests in hand.
- `Measurement.raw.scored_token_ids` records each candidate's selected complete
  path, not discarded alternatives or a complete inference trace. Preview shows
  both alternatives. Do not silently change what these fields claim to expose.
- Every requested backend score must be returned. Missing scores are contract
  errors, not zeros. Do not add a restricted top-K backend to the core.
- Return all exact `top_labels`; random selection is an explicit helper with a
  caller-supplied RNG. Preserve finite log credences through probability underflow.
- Admit exactly `candidate` and `" " + candidate`, with one canonical continuation
  path per surface (SPEC §4.3, §8). Encode whole forms, never toggle whitespace
  between internal tokens or enumerate alternate segmentations. Select by
  first-token logits, preferring the named bare form on exact ties; preserve its
  entire suffix. Reject cross-label path ownership collisions.
- Callers supply optional assistant prefills through `messages=`. Add no answer
  cue. Render native controls and bypass before that prefill, then freeze `C`.
  Continuation encoding must preserve all context tokens and decode to each exact
  surface; handle tokenizer dummy-space rules explicitly or raise. Never trim a
  prefill, repair context by re-encoding it, or silently omit a form.
- `.preview()` must use measurement's exact preparation path, without any backend
  inference. Show full context, visible whitespace, and both possible paths, not
  selected paths or the final readout plan; do not
  silently simplify, truncate, log, or repair the researcher's prompt (SPEC §4.4).

## Defaults and discipline

- Manage the new project as a **uv library**, with `src/credences/`, `py.typed`,
  `pyproject.toml`, and committed `uv.lock`. Use uv for dependencies, environments,
  development commands, and builds (SPEC §12, §15; PLAN Phase 0).
- Planning documents are temporary implementation context, not shipped API
  documentation. Docstrings, code comments, and test explanations must describe
  behavior directly; do not cite `SPEC.md`, `DECISIONS.md`, or `PLAN.md`. References
  to durable package documentation or other code are fine.
- One core operation. Resist adding task verbs, wide rating scales,
  multi-attribute modes, scheduling, or checkpointing (`DECISIONS.md` D2, D12).
- Task-prompt construction, code assignment, option randomization, and semantic
  remapping stay outside the core (`DECISIONS.md` D6). The Credoscope knows the
  candidate strings and two-form policy, not their semantic mappings.
- Broader surface variants and synonym sets are deferred. Beyond the bare and
  one-additional-ASCII-space forms, do not add whitespace, case, or tokenization
  alternatives. Score-based selection is limited to the specified two forms.
- No cross-call inference/KV caching in v1; reuse the loaded model weights.
  A bounded pure plan memo requires identical **selected** paths and `END`.
  Recompute selection per measurement; never cache prompts or first-token scores.
  Optimizations require profiling and parity tests (SPEC §13).
- Measurement calls stay independent and free of cross-call semantic state. Cluster
  workers load one locally available model, build one Credoscope, and process
  independently retryable shards; scheduling and weight staging stay outside the
  package (DECISIONS D12).
- DataFrames stay an optional adapter (`frames.py`), never a core dep.
- Experiment tracking and configuration metadata belong to the researcher, not
  a per-measurement metadata class (`DECISIONS.md` D13). Bypass remains an explicit
  constructor option; do not repeat it on every result.

## Tests

Three tiers (SPEC §14): Hypothesis property tests on the pure core (fake backend);
tokenizer-only boundary/registry tests in ordinary CI; opt-in real-model e2e
behind `CREDENCES_RUN_MLX_TESTS=1` against the blessed-model registry.
**No tests written just to raise the test-coverage number.** Every test names the
failure mode it guards.
If you touched probability math, the property test changes *first*.

## Canonical commands

```bash
uv sync --group dev                         # install (add --extra mlx as needed)
uv run pytest                               # Tiers 1-2 (fast; no model weights)
CREDENCES_RUN_MLX_TESTS=1 uv run pytest       # Tier 3 opt-in real-model e2e
uv run ruff check .
uv run ruff format .
uv run lint-imports                         # dependency-tier check
uv build                                   # build the library distribution
```

## Source-of-truth hierarchy

1. Code + tests define behavior.
2. `SPEC.md` defines intent; `DECISIONS.md` records why.
3. If code and `SPEC.md` disagree, fix one deliberately and note it in the
   changelog — never let them drift silently.

## Before you finish a change

- Dependency tiers intact? (`lint-imports` passes?)
- Public surface still reduces to model loading, build-a-Credoscope, `.measure()`,
  inspection-only `.preview()`, result types, optional tie helpers, and optional
  frame conversion?
- Prefills and task wording stay caller-owned, with only the two starting forms
  supplied by the library and no context rewriting?
- New tests guard a concrete failure mode, not incidental internals?
- Probability math touched ⇒ property test updated first, parity test still green?
- Anything experimental (such as model-profile rendering assumptions) changed ⇒
  changelog entry written?
