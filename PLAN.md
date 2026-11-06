# PLAN.md — building `credences`, step by step

Concrete build order for the new repository. Each phase has a deliverable, a test
gate, and explicit not-yet items. Section references are to `SPEC.md`.

Guiding sequence: pure math → continuation tokenization → one real backend →
complete local instrument → release. Restricted top-K APIs are not a milestone.

---

## Phase 0 — Repository scaffold

**Deliverable:** a public `credences` repository that installs, lints, and runs an
empty test suite.

1. A **uv-managed library project** using `uv init --lib`, Python >= 3.14,
   `src/credences/`, and `py.typed`. Configure the planned Hatchling backend in
   `pyproject.toml`; include MIT license and README.
2. Use uv for dependency/environment management, running tools, and builds.
   Put `ruff`, `pytest`, `hypothesis`, and `import-linter` in the `dev` dependency
   group. Commit `uv.lock`; keep `.venv/` untracked. Runtime integrations remain extras.
3. CI uses `uv sync --locked --group dev` and `uv run` for lint, import boundaries,
   and Python 3.14 tests; opt-in Tier-3 job. Verify `uv build` produces an
   installable library with its typed marker.
4. Copy the planning docs into the repository root.

**Gate:** CI green on the empty package; dependency tiers enforced.

---

## Phase 1 — Pure measurement core

**Deliverable:** `errors.py`, `types.py`, `probabilities.py`, `trie.py`, the
full-logits backend protocol, and a thin fake backend.

1. Result dataclasses: `RawReadout`, `Measurement`, and stable
   JSON-safe `to_dict()` output.
2. Credence math in log space plus entropy, all tied top candidates, and the three
   confidence summaries; optional caller-RNG selection and fractional-count helpers.
3. A pure first-token selector consuming supplied logits: one complete path per
   candidate, with bare selected on exact score ties. Selected-trie construction,
   shared-prefix merging, branch readout plans, and end-vs-continue handling.
   Force singleton transitions; use uniform token fallback for all-`-inf` branches.
   (Shipped as specified at the time; Phase 1b replaces selection with summing.)
4. The `FullLogitsBackend` protocol, with complete requested-score validation.

**Gate — Tier 1:**

- Selection plus trie scoring equals an independent selection and chain-rule
  reference over the selected paths. Include changing branches for the SPEC §8.4
  `positive`/`position` example. Shared first tokens count once, not per candidate.
- Credences sum to 1 and match exponentiated credence logprobs.
- Confidence summaries match their SPEC §6.1 definitions.
- All exact winners are returned without hidden randomness; tie helpers follow
  SPEC §6.4. Finite logs survive probability underflow and JSON serialization.
- End-vs-continue conservation and `END` disjointness, including uniform fallback
  with multiple end-token ids. All-`-inf` is uniform over tokens, not final labels.
- Singleton forcing requires no score query. In mixed readouts, `-inf` tokens get
  zero while finite scores normalize; tiny finite scores never trigger fallback.
- Cross-label path collisions, empty paths, and invalid `END` overlaps raise.
  Preparation checks potential cross-candidate overlaps; scoring uses selected ones.
- Missing requested backend scores and invalid numeric scores raise as specified.
- Golden fixtures cover disjoint single-token candidates, shared-prefix
  multi-token candidates, prefix overlap, and `"A"`/`"AB"` as ordinary candidate
  strings.

**Not yet:** no tokenizer, renderer, model runtime, or real backend.

---

## Phase 1b — Summed spellings in the pure core

**Deliverable:** the pure core matches SPEC §5–§7 as revised: every spelling path
of every candidate in one trie, credences summed per candidate, and coverage.

1. Replace `select_paths` with trie construction over each candidate's one or two
   distinct paths; a path shared by both spellings is stored once; a candidate's
   path that prefixes its own other path raises.
2. Readout plans that always include the scoring position and need no scores to
   build; per-path log probabilities and per-candidate log credences summed in
   log space.
3. `ScoredPath` and the revised `RawReadout` (`scored_paths`, `coverage`) with
   their JSON output; coverage from the scoring position's log normalizer.
4. `FullLogitsBackend.next_token_logits` returns logits plus the full-vocabulary
   log normalizer, validated; update the test fakes.
5. Update docstrings, `docs/pure-core.md`, and the `docs/style.md` example.

**Gate — Tier 1:** the parity property test changes first: an independent
reference enumerates every path and sums each candidate's paths, and keeping only
the stronger spelling must fail it. The §5.2 and §8.4 golden examples match their
hand-computed values. Coverage matches the full-vocabulary probability of the
allowed first tokens and never changes credences.

---

## Phase 2 — Continuations, prefills, and reasoning profiles

**Deliverable:** `tokenization.py` with the SPEC §4.2/§4.3/§8 continuation contract and
`reasoning.py` with `ReasoningProfile`, template introspection, and the
blessed-model registry.

**Gate — Tier 2:**

- Candidates trimmed at the ends. In-context encoding of exactly `candidate` and
  `" " + candidate`, scored from the longest prefix shared with the prompt, with
  exact round trips; no internal whitespace toggling or additional segmentations.
- Native template string/token parity, scalar prompt/user-message equivalence,
  completed few-shot turns, and final assistant prefills. Bypass precedes the
  prefill; no duplicate controls or premature assistant-turn closure.
- Keep prefills exactly, restoring trailing whitespace that Llama 3, Gemma and
  Mistral templates trim. Test no prefill, empty prefill, and cues ending in
  punctuation, a space, or double newlines. No scope-owned answer cue.
- A trailing-space prefill moves the scoring position before the space; candidates
  that merge differently (`"yes"`, `"1"`) share one position. SentencePiece dummy
  spaces never leak into paths. Multibyte text round-trips, or preparation raises
  before inference. No doubled beginning-of-sequence token.
- Path and `END` validation. With fake scores, plan reuse requires identical
  paths, with cold/warm/rebuilt parity.
- Preview report rendering from the same prepared prompt and paths: full text,
  escaped whitespace, where scoring starts, both spellings' token ids grouped by
  candidate, the readout plan, and visible duplicate cues. No model calls,
  automatic output, or separate approximate renderer (SPEC §4.4).
- Exact reasoning-bypass inserts and native-disable kwargs.
- Unlisted-model detection emits its warning.

**Not yet:** nothing executes a model.

---

## Phase 3 — MLX backend, Model, and Credoscope

**Deliverable:** the complete local instrument on Apple Silicon.

1. `backends/mlx.py`: requested final-position logits cast to fp32, plus the
   full-vocabulary log normalizer, internal KV
   reuse for context and trie prefixes only after reviewing request lifetime and
   failure cleanup. Start from an uncached reference. No reasoning generation.
2. `model.py`: `Model` and `load_model`, accepting a registry id or pre-staged
   local snapshot path; backend and reasoning-profile resolution.
3. `credoscope.py`: candidate validation, tokenization, trie execution,
   scalar-or-sequence `measure(prompt=...)` and `measure(messages=...)`, and
   explicit bypass-only reasoning policy. Render the prompt text with an optional
   input prefill, encode both spellings of each candidate in context, build the
   trie plan, read the scoring position and each branch, and sum each candidate's
   paths.
4. `scope.preview(...)`: expose the same preparation without inference, returning
   a readable report or aligned list of reports for the same input shapes.

Before wiring `load_model`, settle backend construction ownership without
violating the import tiers (SPEC §16). Before KV reuse, settle its internal
request lifetime; neither is needed to review the measurement contract.

**Gate:**

- Tier 1 and Tier 2 remain green through the public Credoscope surface.
- Bare strings as candidate collections, mappings, duplicates, empty candidates,
  fewer-than-two candidates, and candidates duplicated after trimming are
  rejected at construction.
- Scalar input returns `Measurement`; sequence input preserves order and length;
  empty sequences return `[]`. A failing element raises without partial results.
- Preview/measurement preparation and input-validation parity; preview shows
  the paths and plan measurement uses, makes zero backend calls, and interleaved
  previews do not change results.
- Consecutive calls do not influence one another; independent shards produce the
  same results within backend numerical tolerance as inputs measured together.
- Results contain neither run metadata nor a redundant reasoning field;
  `raw.scored_paths` exposes every scored path with its probability, and
  coverage and context length are always present.
- Opt-in real-model tests cover one non-reasoning model, one `</think>` family,
  and one Harmony family. Unsupported reasoning values raise at build time.
- A real-model multi-token result matches independent naive masked scoring.
- Loading from a complete local snapshot requires no network access.

**Not yet:** no hosted API, prompt renderer, presentation randomization, semantic
mapping, scheduler, or cross-call inference/KV cache. Pure trie-plan reuse is allowed.

---

## Phase 4 — Frames adapter, docs, and 0.1.0

1. Optional `frames.py` adapter for `list[Measurement]`.
2. README quickstart with preview-before-measurement, forced-choice semantics,
   reasoning caveats, confidence summaries, and citations. Spelling and prefill
   details stay in `docs/tokenization.md`.
3. Deployment note: independently retryable shards, one model load per worker,
   and weights pre-staged in shared or node-local storage.
4. Publish `credences==0.1.0` after a clean-environment quickstart succeeds.

---

## Deferred beyond 0.1.0

- Standalone prompt rendering, option-order randomization, balanced code
  assignment, and candidate-to-semantic-label metadata; see
  `POSSIBLE_FUTURE_EXTENSIONS.md`.
- Spellings beyond bare/one-leading-ASCII-space, caller-listed spellings (such as
  capitalization) and synonym sets, and non-canonical tokenizations.
- Generated reasoning modes, including post-trace measurement and multi-sample
  averaging; see `POSSIBLE_FUTURE_EXTENSIONS.md` §8.
- A second full-access backend, including NVIDIA or a researcher-controlled server.
- Restricted probability APIs only for a concrete future use case; no placeholder
  protocol or provider implementation in the core. See future extensions §9.
- Cross-call inference-state caching, only after profiling; loaded weights are reusable.

## Standing risks

- Token-boundary errors can silently corrupt every value; every supported model
  needs Tier-2 coverage.
- Reasoning bypass strings are model-version-specific and require registry tests.
- Native openings, explicit prefills, and canonical suffix encoding must be
  tested with each supported tokenizer; fail instead of rewriting context or
  silently dropping an unsupported form.
- Caller-owned semantic mappings must travel with stable input IDs in distributed
  outputs so candidate credences cannot be attached to the wrong labels.
