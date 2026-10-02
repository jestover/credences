# DECISIONS.md — why `credences` is shaped this way

Records the load-bearing decisions and their rationale. Mechanics and contracts
live in `SPEC.md`; working rules live in `AGENTS.md`. Read `SPEC.md` §1–§8 first,
then this file, then `AGENTS.md`.

**CDS** = Chen, Didisheim & Somoza, *Out of the Black Box* (NBER WP 34965,
2026), whose forced-choice logprob method and binary "inner confidence" metric
help situate this tool in the existing literature.

## D1. One purpose: extract credence distributions

`credences` extracts a model's probability distribution over a fixed set of
candidate answers. Domain logic and task-prompt construction belong in consuming
projects. → SPEC §1, §3.

## D2. One core operation

No `classify`/`rate`/`label` verbs: they differ only in prompt wording and output
shape. A Credoscope measures candidate answers supplied by the caller. → SPEC §3,
§4.

`.preview()` is a no-inference inspection helper, not another measurement mode.
It shares preparation with `.measure()` and returns a readable report of the
actual context and candidate continuations. Users can inspect role delimiters,
duplicate cues, whitespace, and both possible token paths without inference or
prompt metadata on every result. It cannot identify selected paths without the
model's first-token scores. The package does not guess which wording to remove.
→ SPEC §4.4.

## D3. Bind exact candidate strings once

`build_credoscope` binds a model to a `Sequence[str]` of exact answer strings.
The candidates may be semantic words such as `"positive"` or codes such as `"A"`,
but the Credoscope does not distinguish those cases. It has no descriptions,
semantic-label mapping, or presentation metadata. At least two unique, non-empty
candidates are required. Each has two canonical continuation forms: the exact
candidate and one additional ASCII space followed by it. Internal spelling and
whitespace stay fixed; no other forms or segmentations are added. First-token
scores select one complete form per candidate, with bare preferred on exact ties.
Result keys remain the original strings. Reject overlapping accepted surface
sets or token paths across labels. → SPEC §4.1, §4.3, §8.

## D4. Accept one input or a sequence

`.measure()` accepts one complete prompt or chat specification and returns a
`Measurement`, or accepts a sequence and returns `list[Measurement]` in input
order. The scalar form keeps interactive use direct; the sequence form provides a
stable surface for future backend batching without a second measurement verb. → SPEC
§4.2, §6.3.

## D5. The measurand is the forced-choice distribution

First-token scores choose one representation per candidate; constrained scoring
then uses only those selected paths. Shared prefixes receive one weight before
later branches divide it. Do not sum the forms or normalize per-label maxima
before merging shared first tokens. This is the distribution of a decoder using
the same selection, stopping, and transition rules, not every structured-output
engine or global conditioning on complete valid answers. Selection plus trie
scoring must match an independent two-stage reference. → SPEC §5, §7, §14.

Singleton transitions are forced with probability 1. At an all-`-inf` branch,
use uniform probability over distinct allowed token ids; when any score is finite,
`-inf` tokens get zero. The fallback completes the procedure without inventing a
preference among tokens, but is not evidence of equal label beliefs. Structural
validation and missing/NaN/positive-infinity score errors remain errors.

## D6. Task-prompt construction and semantic mappings are external

The caller owns prompt wording, option order, code assignment, and any mapping
from candidate strings to semantic labels. Keeping the mapping inside the
Credoscope would duplicate the renderer's source of truth without letting the
Credoscope verify what the prompt actually says. Coded results should be persisted
with a stable input identifier and the per-input candidate-to-label mapping.
A possible standalone rendering helper is deferred. → SPEC §3, §11;
`POSSIBLE_FUTURE_EXTENSIONS.md`.

The caller also owns any assistant prefill, supplied as the final assistant
message. A chat `prompt=` is simply one user message, not a second prefill
interface. The library supplies native chat formatting and reasoning bypass,
but no answer cue or formatting template. This keeps user instructions and
assistant content in their declared roles. → SPEC §4.2, §8.

## D7. One full-logits capability contract

`FullLogitsBackend` scores arbitrary requested token ids at supplied contexts,
using an accessible, matching tokenizer. V1 implements MLX. Future cluster or
researcher-controlled server runtimes may implement the same contract; physical
locality is not the requirement. Requested logits are enough for masked scoring,
so the protocol need not return a full-vocabulary probability array. → SPEC §10.

## D8. Restricted probability APIs are outside the core

Do not shape the core around truncated scores, opaque tokenization, or a lack of
continuation access. There is no top-K protocol, provider-specific result shape,
or hosted-backend release gate. A short future note preserves the constraints for
a concrete use case later. Missing requested scores in a full-logits backend are
contract errors, never probability zero. → SPEC §10;
`POSSIBLE_FUTURE_EXTENSIONS.md` §9.

## D9. Bypass is the only reasoning mode in v1

Keep `reasoning="bypass"` explicit, using a native disable switch or an empty
thinking phase. No generated reasoning, per-call mode overrides, or token-cap
machinery or per-result reasoning field in v1. Bypass and conditioning on a
generated thought trace measure different contexts; neither is inherently more
accurate. Learnings for a future
comparison live in `POSSIBLE_FUTURE_EXTENSIONS.md` §8. → SPEC §9.

## D10. Use “credence” for the distribution

A credence is a degree of belief. “Confidence” is overloaded and usually names a
scalar, while this package returns the complete distribution. Scalar confidence
values are explicitly derived summaries. → SPEC §2, §6, §15.

## D11. Confidence summaries are derived from the vector

The generic summaries are `top_label_confidence`, `entropy_confidence`, and
`margin_confidence`. Each follows from the recovered distribution without adding
structure to the candidates. Ordered or asymmetric applications require
researcher-supplied semantic mappings, geometry, or losses and remain downstream
projections. → SPEC §6.

`top_labels` exposes all exact maximizers without hidden tie-breaking. Optional
helpers provide caller-controlled random selection or fractional winner counts.
Entropy is in bits; entropy confidence provides the normalized alternative.
Log credences preserve tiny values even when ordinary probabilities underflow.

## D12. Distribution belongs outside the package

Measurement calls are independent, deterministic on deterministic backends, and
free of cross-call semantic state. Reusing an immutable trie plan does not retain
model state or change the measurement. A cluster runner should divide a corpus into
small, independently retryable shards, load the model once per worker, and process
many inputs. Model weights should be pre-staged in a shared cache or copied once
to node-local storage. Scheduling, preemption, retries, and checkpoint manifests
belong to SLURM or the consuming application. → SPEC §4, §13.

## D13. Experiment metadata belongs to the researcher

The core does not manage experiment provenance or repeat configuration metadata
on every measurement. Results contain the vector, summaries, and raw readout;
the researcher decides how to record model versions, prompts,
code, and run metadata. Independent measurement calls are an execution contract,
not a promise of bitwise reproducibility across environments. → SPEC §5.5, §6.

## D14. Two starting forms at a fixed context

Bare and leading-space forms can have different relative label probabilities.
Use the model's first-token logits at each prompt to choose a form per candidate,
then retain its entire path. This avoids scoring both later subtrees, while making
the representation policy explicit. It is not summation across spellings or
maximum complete-answer likelihood; discarded forms need not be noise or have
negligible mass. Restricting each surface to canonical continuation encoding
bounds work without enumerating every possible segmentation.

Only the start of the whole candidate varies. Tokenize `c` and `" " + c` in full;
do not assume their tokenizations differ only in the first token, and do not
toggle whitespace at internal token boundaries. Extra newlines, tabs, case
variants, synonyms, and all-tokenization marginalization are outside v1.

Render the native assistant opening, bypass, and optional caller prefill first,
then freeze every context token. Preserve a prefill's trailing whitespace; the
spaced form adds one more space even when one is already present. Canonical
continuation encoding must respect that boundary rather than re-tokenizing the
whole concatenated text and absorbing a prompt token. Tokenizer-specific
start-of-string preprocessing requires tested suffix behavior, not guesswork.

The scoring trie is compiled after the first-token readout, and may be reused only
for identical selected paths and `END`. Selection runs again for every measurement.
Preview shows the fixed context and both alternatives, not a selected plan.
Small score changes near a form tie can change later branches; deterministic does
not mean insensitive to numerical variation. → SPEC §4.3, §4.4, §8.

## D15. uv-managed library project

Use uv's library template and a `src/credences/` layout with `py.typed`, not an
application/script scaffold. uv manages dependencies, the development environment,
committed lockfile, commands, and builds. Keep the planned Hatchling build backend;
developer tools use a `dev` dependency group and runtime integrations use optional
extras. This is project tooling, not a requirement that downstream users run uv.
→ SPEC §12, §15; PLAN Phase 0.
