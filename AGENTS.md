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
  first** (SPEC §14). Score every spelling path of every candidate in one trie;
  a candidate's credence is the sum of its paths. Never keep only one spelling.
  Merge shared prefixes before normalization; never count a shared first token
  once per label or per spelling, and count a path shared by both spellings once.
- Force singleton transitions to probability 1 without querying them. All-`-inf`
  branches use uniform probability over distinct allowed token ids. With any
  finite score, `-inf` tokens get zero. Missing/NaN/positive-infinity scores remain
  errors; fallback is not a repair for malformed backend output.
- The answer boundary (SPEC §8) and reasoning inserts (SPEC §9.1) are the two
  easiest places to silently corrupt every number. Touch them only with Tier-2
  tokenizer tests in hand.
- `Measurement.raw.scored_paths` records every scored path with its spellings,
  tokens, and probability; the paths sum to the credence. Preview shows the same paths
  and the readout plan. Do not silently change what these fields claim to expose.
- Every requested backend score must be returned. Missing scores are contract
  errors, not zeros. Do not add a restricted top-K backend to the core.
- `raw.coverage` is a diagnostic only. Never let it change credences, filter or
  reweight results, or trigger warnings; low coverage is not low accuracy.
- Return all exact `top_labels`; random selection is an explicit helper with a
  caller-supplied RNG. Preserve finite log credences through probability underflow.
- Trim candidates at both ends only; keep internal whitespace; keep result keys
  as supplied. Score exactly `candidate` and `" " + candidate`, each in its one
  canonical tokenization (SPEC §4.1, §4.3). Never toggle whitespace between
  internal tokens, add case or newline variants, or add non-canonical
  tokenizations. Reject cross-label path ownership collisions.
- Encode each spelling together with the prompt text, never on its own; score
  from the longest token prefix shared by the prompt and every spelling, and
  require exact round trips (SPEC §8.2). Never encode rendered chat text with
  automatic special tokens.
- Callers supply optional assistant prefills through `messages=`. Add no answer
  cue. Render native controls and bypass before that prefill. Prompt and prefill
  text are never changed: restore trailing whitespace a template trims, and
  raise on any other template change. The token boundary may move to the shared
  prefix; the text may not. Never silently omit a spelling.
- `.preview()` must use measurement's exact preparation path, without any backend
  inference. Show the full prompt, visible whitespace, where scoring starts, every
  scored path, and the readout plan; do not silently simplify, truncate, log, or
  repair the researcher's prompt (SPEC §4.4).

## Defaults and discipline

- Manage the new project as a **uv library**, with `src/credences/`, `py.typed`,
  `pyproject.toml`, and committed `uv.lock`. Use uv for dependencies, environments,
  development commands, and builds (SPEC §12, §15; PLAN Phase 0).
- Planning documents are temporary implementation context, not shipped API
  documentation. Docstrings, code comments, and test explanations must describe
  behavior directly; do not cite `SPEC.md`, `DECISIONS.md`, or `PLAN.md`. References
  to durable package documentation or other code are fine.
- Use **Google-style docstrings** with `Args`, `Returns`/`Yields`, `Raises`,
  `Attributes`, and `Examples` sections where applicable. They target API users
  who have never read this file or `SPEC.md`: plain words, terms defined on
  first use, no internal shorthand (see "Write documentation for a newcomer" in
  `docs/style.md`). Explain purpose, input/output meaning, and representative
  usage, not constructor or loop mechanics. The README follows the same rule;
  contributor process stays here. Types live in annotations; prose adds semantics and constraints.
  Non-obvious implementation rationale belongs in sparse code comments; avoid
  line-by-line narration and comment tutorials.
- Include small runnable examples for non-obvious core behavior, and a shared
  walkthrough when several objects interact. Keep examples checked by the normal
  test run so they stay useful as the code evolves. Round display output only.
- One core operation. Resist adding task verbs, wide rating scales,
  multi-attribute modes, scheduling, or checkpointing (`DECISIONS.md` D2, D12).
- Task-prompt construction, code assignment, option randomization, and semantic
  remapping stay outside the core (`DECISIONS.md` D6). The Credoscope knows the
  candidate strings and two-spelling policy, not their semantic mappings.
- Broader spelling variants and synonym sets are deferred. Beyond the bare and
  one-additional-ASCII-space spellings, do not add whitespace, case, or
  tokenization alternatives. A pruning tolerance may only approximate the fixed
  measurand with bounded error, never decide which spellings count.
- No cross-call inference/KV caching in v1; reuse the loaded model weights.
  A bounded pure plan memo requires identical paths and `END`; never cache
  prompts, scores, or logits. Optimizations require profiling and parity tests
  (SPEC §13).
- Measurement calls stay independent and free of cross-call semantic state. Cluster
  workers load one locally available model, build one Credoscope, and process
  independently retryable shards; scheduling and weight staging stay outside the
  package (DECISIONS D12).
- DataFrames stay an optional adapter (`frames.py`), never a core dep.
- Experiment tracking and configuration metadata belong to the researcher, not
  a per-measurement metadata class (`DECISIONS.md` D13). Bypass remains an explicit
  constructor option; do not repeat it on every result.

## Code style and static checks

Follow `docs/style.md` and the explicit settings in `pyproject.toml`.

- Ruff formatting is authoritative: four spaces, double quotes, and a 100-column
  target, including formatting code examples in docstrings.
- Use PEP 8 naming, sorted imports, and modern Python 3.14 syntax. Do not enable
  `ALL` or rewrite clear probability math to satisfy speculative style preferences.
- Annotate library functions and methods. `uv run ty check` checks `src/`;
  warnings and unused ignores fail. Fix inaccurate types rather than masking
  diagnostics with blanket ignores or `Any`.
- Tests keep ordinary lint checks but are exempt from required annotations and
  docstrings. Deliberately invalid test inputs stay outside the default ty target.
- Static checks do not replace runtime score validation, property tests, or
  tokenizer boundary tests.
- Check data once, where it enters the package: `build_trie` checks token paths
  and `TriePlan.score` checks every backend score before using any. Helpers
  behind those entry points start with an underscore, live in the module that
  uses them, and trust their inputs; their types and docstrings say what they
  require and what goes wrong otherwise. Do not re-check the package's own
  arithmetic; keep only guards that stop a bound or clamp from hiding a bug.

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
uv run ty check                            # shipped-library type check
uv run lint-imports                         # dependency-tier check
uv build                                   # build the library distribution
```

## Publishing and primary-checkout synchronization

Keep development isolated in the thread checkout. The user's original checkout
is a clean review/run copy of the published `main` branch.

At a completed parent-thread step:

1. Validate locally, commit, and push `main` to the configured GitHub upstream.
   Inspect remotes first; `local` is the primary-checkout backlink, not GitHub.
2. Verify GitHub CI passed for the exact published commit before updating the
   original checkout.
3. Resolve the original checkout from `local` and verify it is the intended
   checkout on this machine, on `main`, with no tracked or untracked changes.
   Stop and report if the target is unexpected, dirty, missing, or divergent.
4. Fast-forward it to the published commit using the configured `local` remote
   or an ordinary GitHub fetch/fast-forward. Never force-push, reset away edits,
   or copy/delete files to make synchronization succeed.
5. Refresh its `origin/main` reference and verify the thread, GitHub `main`, and
   original `main` all match. Preserve ignored local files and backup branches;
   do not publish local-only backups.

For changes published from elsewhere, the original checkout updates with
`git pull --ff-only`. Its local-only `backup/original-scaffold` branch preserves
the files present before the initial synchronization.

Worker and review subagents do not publish or synchronize the primary checkout
unless explicitly assigned that task. This is a publishing routine, not a
background watcher for pushes made elsewhere.

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
- Prefills and task wording stay caller-owned and unchanged, with only end
  trimming of candidates and their two spellings supplied by the library?
- New tests guard a concrete failure mode, not incidental internals?
- Probability math touched ⇒ property test updated first, parity test still green?
- Anything experimental (such as model-profile rendering assumptions) changed ⇒
  changelog entry written?
