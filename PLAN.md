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

## Phase 2 — Continuations, prefills, and reasoning profiles

**Deliverable:** `tokenization.py` with the SPEC §4.2/§4.3/§8 continuation contract and
`reasoning.py` with `ReasoningProfile`, template introspection, and the
blessed-model registry.

**Gate — Tier 2:**

- Fixed-context canonical encoding of exactly `candidate` and `" " + candidate`;
  both complete forms are validated, without internal whitespace toggling or
  enumeration of additional segmentations.
- Native template string/token parity, scalar prompt/user-message equivalence,
  completed few-shot turns, and final assistant prefills. Bypass precedes the
  prefill; no duplicate controls or premature assistant-turn closure.
- Preserve trailing prefill whitespace. Test no prefill, empty prefill, and cues
  ending in punctuation, a space, or double newlines. No scope-owned answer cue.
- Handle tokenizer dummy initial spaces in a tested suffix encoder. Exact
  context-plus-path decoding, multibyte text, and fixed-context whitespace all
  work, or raise before inference; never absorb a context token into a path.
- Context-dependent potential-path and `END` validation. With fake scores,
  selected-plan reuse requires identical selected paths, with cold/warm/rebuilt
  parity and a regression for identical alternatives but different selections.
- Preview report rendering from the same prepared context and paths: full text,
  escaped whitespace, both forms' token ids grouped by candidate, and visible
  duplicate cues. No selected-path claims, model calls, automatic output, or
  separate approximate renderer (SPEC §4.4).
- Exact reasoning-bypass inserts and native-disable kwargs.
- Unlisted-model detection emits its warning.

**Not yet:** nothing executes a model.

---

## Phase 3 — MLX backend, Model, and Credoscope

**Deliverable:** the complete local instrument on Apple Silicon.

1. `backends/mlx.py`: requested final-position logits cast to fp32, internal KV
   reuse for context and trie prefixes only after reviewing request lifetime and
   failure cleanup. Start from an uncached reference. No reasoning generation.
2. `model.py`: `Model` and `load_model`, accepting a registry id or pre-staged
   local snapshot path; backend and reasoning-profile resolution.
3. `credoscope.py`: candidate validation, tokenization, trie execution,
   scalar-or-sequence `measure(prompt=...)` and `measure(messages=...)`, and
   explicit bypass-only reasoning policy. Render fixed context with an optional
   input prefill, compile both canonical forms, query distinct first ids, and
   select one whole path per candidate. Build and score the selected trie, reusing
   initial scores for its root and querying only its later branch points.
4. `scope.preview(...)`: expose the same preparation without inference, returning
   a readable report or aligned list of reports for the same input shapes.

Before wiring `load_model`, settle backend construction ownership without
violating the import tiers (SPEC §16). Before KV reuse, settle its internal
request lifetime; neither is needed to review the measurement contract.

**Gate:**

- Tier 1 and Tier 2 remain green through the public Credoscope surface.
- Bare strings as candidate collections, mappings, duplicates, empty candidates,
  fewer-than-two candidates, and overlapping accepted surface sets are rejected
  at construction.
- Scalar input returns `Measurement`; sequence input preserves order and length;
  empty sequences return `[]`. A failing element raises without partial results.
- Preview/measurement preparation and input-validation parity; preview shows
  both alternatives without predicting selection, makes zero backend calls,
  and interleaved previews do not change results.
- Consecutive calls do not influence one another; independent shards produce the
  same results within backend numerical tolerance as inputs measured together.
- Results contain neither run metadata nor a redundant reasoning field; raw
  token ids expose the selected complete path per candidate, and context length
  is always known.
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
   two-form/prefill and reasoning caveats, confidence summaries, and citations.
3. Deployment note: independently retryable shards, one model load per worker,
   and weights pre-staged in shared or node-local storage.
4. Publish `credences==0.1.0` after a clean-environment quickstart succeeds.

---

## Deferred beyond 0.1.0

- Standalone prompt rendering, option-order randomization, balanced code
  assignment, and candidate-to-semantic-label metadata; see
  `POSSIBLE_FUTURE_EXTENSIONS.md`.
- Forms beyond bare/one-leading-ASCII-space, synonym sets, and enumeration of all
  tokenizations of an accepted text.
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
