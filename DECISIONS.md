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
actual prompt, every scored token path, and the readout plan. Users can inspect
role delimiters, duplicate cues, whitespace, and where scoring starts without
inference or prompt metadata on every result. Because no step depends on the
model's scores, preview shows exactly what measurement will use. The package does
not guess which wording to remove. → SPEC §4.4.

## D3. Bind candidate strings once, trimmed at the ends

`build_credoscope` binds a model to a `Sequence[str]` of answer strings. The
candidates may be semantic words such as `"positive"` or codes such as `"A"`,
but the Credoscope does not distinguish those cases. It has no descriptions,
semantic-label mapping, or presentation metadata. At least two candidates are
required.

Leading and trailing whitespace is trimmed before scoring; internal whitespace is
kept. End whitespace is about where an answer sits, which the two spellings
already measure (D14): a candidate written `" yes"` would double the spaced
spelling's space and remove the unspaced option. Internal whitespace is part of
what the answer says, so collapsing it would silently change the scored text and
could merge distinct candidates. Most users never think about this, so the rule
is stated plainly in the documentation. Result keys remain the original strings
so lookups keep working; candidates that are empty or duplicated after trimming
are rejected. → SPEC §4.1, §4.3, §8.

## D4. Accept one input or a sequence

`.measure()` accepts one complete prompt or chat specification and returns a
`Measurement`, or accepts a sequence and returns `list[Measurement]` in input
order. The scalar form keeps interactive use direct; the sequence form provides a
stable surface for future backend batching without a second measurement verb. → SPEC
§4.2, §6.3.

## D5. The measurand is the forced-choice distribution over both spellings

A candidate's credence is the probability that a decoder, allowed to write only
the candidates' spellings, writes that candidate with or without the leading
space. All paths share one trie; shared prefixes receive one weight before later
branches divide it, and each candidate sums its paths. This is the distribution
of a decoder using the same masking, stopping, and transition rules, not every
structured-output engine or global conditioning on complete valid answers. Trie
scoring must match an independent reference that enumerates and sums every path.
→ SPEC §5, §7, §14.

Summing replaced an earlier rule that kept only the spelling with the higher
first-token score. The case for choosing was real: the model almost always
prefers one spelling, so summing rarely moves a credence much (at most 0.0014 in
SPEC §5.2's example), and choosing looks like a cleaner decision. Summing won
because the allowed texts are identical under both rules, so it opens no door to
further variants; it removes a discontinuity, since choosing jumps between two
continuations when two first-token scores cross; it costs nothing for one-token
candidates and only extra branch readouts where candidates share a first word
under the weaker spelling; and "with or without a space" is easier to explain
than "whichever first token scored higher, bare on ties".

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
so the protocol need not return a full-vocabulary probability array. Each readout
also returns one number, the full-vocabulary log normalizer, used only for the
coverage diagnostic (D16). → SPEC §10.

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

## D14. Two spellings, scored in context

Each candidate is scored as written and after one space. Most tokenizers fold
the space between words into the following word's token, and whether a model
writes it depends on what precedes the answer: chat headers ending in a newline
suggest the bare spelling, while `"Answer:"` or Mistral's `[/INST]` suggest the
spaced one. Scoring both avoids guessing per model. Anything else in front of the
answer (a newline, an extra space, markdown) is the model choosing to format its
reply, like starting with "The", and stays excluded. Capitalization and synonyms
change the word, so they remain the caller's choice of candidates.

Each spelling uses one canonical tokenization. Geh et al. (EMNLP 2024) show that
the most likely tokenization and the sum over all tokenizations are NP-hard and
#P-hard to compute, that the canonical tokenization usually holds nearly all of
a text's probability (about 99.6% for `Tokens` under Llama 2), and that gains from
non-canonical tokenizations needed a mixing weight tuned on held-out data. A
measurement should not depend on such tuning, so non-canonical tokenizations of
a spelling are not added.

Spellings are encoded together with the prompt text, and scoring starts at the
longest token prefix shared by the prompt and every spelling. Tests on nine
tokenizer families showed why: SentencePiece tokenizers (Mistral, Phi) give an
answer encoded on its own a dummy space, so the bare spelling silently became
the spaced token; a prompt ending in a space leaves a lone space token that
byte-level tokenizers would have merged into the answer word; and newlines can
re-split before a word. Moving the start of scoring to the shared prefix gives
the model the tokens it would see for the full text, without changing any of
the user's text. lm-evaluation-harness does a narrower version of this (moving a
context's trailing whitespace into the continuation); computing the shared prefix
also catches non-whitespace merges and verifies the split. An exact round trip
of every encoding is required, so a tokenizer that alters text fails loudly.

Prefills are kept exactly, including trailing whitespace, because they are text
the user chose to have the model "already write". Some templates (Llama 3, Gemma,
Mistral) trim them, and `transformers` follows suit silently, so credences
restores the trimmed whitespace and raises on any other change. A trailing space
then needs no special handling: it merges into the answer tokens. Prefills stay
allowed because they often move a chat model straight to its answer.

The scored paths and readout plan depend only on the prompt and candidates, not
on scores, so plans may be reused for identical paths and `END`, and preview can
show the plan. → SPEC §4.3, §4.4, §8.

## D15. uv-managed library project

Use uv's library template and a `src/credences/` layout with `py.typed`, not an
application/script scaffold. uv manages dependencies, the development environment,
committed lockfile, commands, and builds. Keep the planned Hatchling build backend;
developer tools use a `dev` dependency group and runtime integrations use optional
extras. This is project tooling, not a requirement that downstream users run uv.
→ SPEC §12, §15; PLAN Phase 0.

## D16. Coverage is a diagnostic, not a quality score

`raw.coverage` reports the model's unrestricted probability that its first
answer token begins an allowed spelling. It does not enter credences, and the
library never filters, reweights, or warns on it, because low coverage is not
evidence of worse credences. A model that habitually opens with "The"
(`"The"` 0.90, `"yes"` 0.07, `"no"` 0.02) and one that answers directly
(`"yes"` 0.70, `"no"` 0.20, `"The"` 0.01) produce the same credences, yes 0.78 and
no 0.22, at coverage 0.09 and 0.90. Which is more accurate is an empirical,
model-dependent question that coverage lets researchers study. It is a small raw
field, not a headline feature. → SPEC §6.5, §10.2.
