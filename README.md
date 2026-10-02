# credences

LLM credence vectors over fixed answer sets.

**Status:** library scaffold and agreed design documents. The measurement API is
not implemented yet. Start implementation from `PLAN.md`, not the illustrative
API examples as though they already exist.

## Development setup

This is a Python >= 3.14 library, initialized with `uv init --lib` and built with
Hatchling. See [uv's library guide](https://docs.astral.sh/uv/concepts/projects/init/#libraries).
Development uses Python 3.14.8, pinned in `.python-version` and managed by uv.

```bash
uv sync --locked --group dev
uv run ruff check .
uv run ruff format --check .
uv run lint-imports
uv run pytest
uv build
```

The `dev` group includes pytest, Hypothesis, Ruff, and import-linter. The package
has no runtime dependencies yet; model and DataFrame integrations will be optional
extras as they are implemented. No model weights are needed for this scaffold.

The test suite is currently empty, so `uv run pytest` reports no tests (exit 5).
The GitHub Actions workflow runs the checks above and temporarily accepts that
specific exit code; other pytest failures still fail CI. Remove this allowance
when the first contract tests are added.

CI also installs the built wheel into an isolated environment and checks that it
imports, includes `py.typed`, and has no runtime dependencies. Import-linter is
configured to inspect the package, but no dependency-tier contracts exist yet:
add them as the planned modules are introduced. The opt-in MLX job will follow
when real-model tests and the MLX extra exist; no model weights are downloaded now.

No measurement behavior or placeholder public API has been added. The next small
implementation step is Phase 1's result objects and serialization contract.

## Design documents

These files describe the intended `credences` package and are the starting context
for implementation in this repository.

These docs define the intended package, not an already implemented API.
`DECISIONS.md` explains the rationale for the design.

## Current core

- Arbitrary-token logit access at supplied contexts; MLX is the first backend.
  Restricted top-K probability APIs are outside the core, not a release milestone.
- Two canonical forms per candidate: bare and one additional ASCII space.
  First-token logits select one complete path per candidate; score only the
  selected trie, merging shared prefixes. Exact form-score ties select bare.
- Forced transitions have probability 1. At a branching node, an all-`-inf`
  readout falls back to uniform over allowed tokens; otherwise `-inf` tokens
  get zero and finite scores normalize normally.
- Native assistant opening with no automatic answer cue. Optional assistant
  prefills are supplied through `messages=`; all context tokens stay fixed.
  Task wording, code meanings, and option randomization stay external.
- `print(scope.preview(prompt=...))` shows the full compiled context and each
  candidate's possible continuations, including visible whitespace and token paths, without
  inference. It cannot predict which form the score readout will select.
- Explicit `reasoning="bypass"` at construction; no redundant reasoning or run
  metadata fields on each measurement.
- Independent measurements; reusable pure trie plans, but no model state carried
  between calls. Loaded weights remain reusable. Corpus sharding and model-weight
  staging remain external.
- A uv-managed library project with `src/credences/`, `py.typed`, `pyproject.toml`,
  and committed `uv.lock`. Use uv for dependencies, environments, checks, and builds.

The optional space is only at the start of the complete candidate, not before
each token. Encode both whole forms canonically; do not enumerate other
segmentations, case variants, or additional whitespace. For example,
`"very positive"` may have paths `["very", " positive"]` and
`[" very", " positive"]`, but not `["very", "positive"]`. Select by first-token
score, preserving the selected form's entire suffix. Tokenization can change more
than the first piece: `"positive"` may be `["pos", "itive"]` while `" positive"`
is one token. Do not find alternatives by regex-matching only the first piece.

The selected paths determine branch points. With `positive` and `position`, both
bare paths might start with `"pos"`; that prefix needs a later comparison only if
both paths are selected. A shared first token receives one root weight, not one
per label. See the worked example in SPEC §8.4. Selection is not summing or
averaging both forms, and a token-uniform fallback need not be label-uniform.

Prefill whitespace is literal: `"Answer: "` stays in the fixed context, so the
two forms produce one or two spaces before the label. This is a specified
token-trie constraint, not a claim to capture every possible answer spelling or
to remove prompt sensitivity. Different token granularities can also affect the
locally masked distribution (SPEC §5.5).

Read `SPEC.md` §4.2-§4.4 and §8 together for input handling, starting forms, and
preview. Preview shares measurement's inference-free preparation and displays
both alternatives without rewriting context; `raw.scored_token_ids` records the
one selected path per candidate after measurement. Model loading ownership,
backend request lifetime, and supported-tokenizer validation remain implementation checkpoints
(§16), not reasons to postpone this review.

## What each file is for

- `confidence-notes.typ` is the conceptual note: why the credence vector is the
  primitive object and why the confidence summaries are derived from it.
- `confidence-notes.pdf` is rendered output from `confidence-notes.typ`; do not
  treat it as a separate source.
- `SPEC.md` is the normative contract for v1. It defines the API, measurement
  semantics, result objects, backend contract, architecture, and tests.
- `DECISIONS.md` records why the design is shaped this way. It should explain the
  load-bearing choices without duplicating the mechanics from the spec.
- `PLAN.md` turns the spec into an implementation sequence with phase gates.
- `AGENTS.md` is the working agreement for coding agents and humans implementing
  the package.
- `POSSIBLE_FUTURE_EXTENSIONS.md` is a parking lot for useful additions that
  should not enter the minimal core, including a standalone prompt renderer with
  presentation randomization and semantic-mapping metadata, plus generated
  reasoning modes beyond v1's explicit `reasoning="bypass"`.

## Suggested review order

1. `confidence-notes.typ` for the conceptual frame.
2. This `README.md` for the map of the docs.
3. `SPEC.md` sections 1-8 for the core contract: purpose, API shape, measurand,
   result object, trie semantics, and the continuation-tokenization walkthrough.
4. `DECISIONS.md` for the rationale behind the contract.
5. `SPEC.md` sections 9-16 for reasoning, backend capabilities, prompt ownership,
   architecture, testing, distribution, and open questions.
6. `PLAN.md` to check that the project can be built in a sensible order.
7. `AGENTS.md` to review implementation discipline and source-of-truth rules.
8. `POSSIBLE_FUTURE_EXTENSIONS.md` after the v1 shape is clear.

Readers who already know the conceptual note can start with this README, then
follow steps 3-8. For implementation work, read `SPEC.md`, `DECISIONS.md`, and
`PLAN.md` before writing code.
