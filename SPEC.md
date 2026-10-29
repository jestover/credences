# credences — Specification (v1)

> **Status:** seed specification for a fresh `credences` repository.
>
> **Reading order:** read this file §1–§8 first. Then `DECISIONS.md`
> for *why* the design is shaped this way, then `AGENTS.md` for the working rules.
> **Review focus:** each candidate is scored in two spellings, as written and
> after one space, and its credence is their combined probability (§4.3, §5.1).
> Both spellings are encoded together with the prompt text and scored from the
> tokens they share with it (§8). The caller owns optional assistant prefills
> through `messages=`; prompt and prefill text are never changed.
> V1 requires arbitrary-token logit access and implements MLX only (§10).
> Model-specific rendering and tokenization must pass the §14 tests before a
> model is supported. Remaining implementation questions are listed in §16.

## 1. What `credences` is

`credences` is a small Python library that does **one thing**: given a prompt and a
small set of candidate answers, it returns the model's **credence vector** — its
probability distribution — over those answers, read directly from the model's
next-token probabilities.

A *credence* is a degree of belief: a subjective probability in [0, 1] (the standard
term in Bayesian epistemology; Titelbaum, *Fundamentals of Bayesian Epistemology I*,
OUP 2022). When a language model is asked a forced-choice question, its next-token
distribution under the specified prompt, answer representation, and masking rule
is the operational definition of its credence here. This is not a claim of
calibration or a prompt-independent belief. The vector is the primitive output.
Top labels and scalar quantities such as top-label confidence, entropy confidence,
and margin confidence are summaries **of** that vector, not replacements for it.

```python
from credences import load_model, build_credoscope

model = load_model("mlx-community/Qwen3-4B-4bit")           # local model via MLX
scope = build_credoscope(
    model,
    candidates=["positive", "negative", "neutral"],
)

prompt = ("Classify the sentiment of: 'Revenue beat expectations.' "
          "Answer with exactly one of: positive, negative, neutral.")
print(scope.preview(prompt=prompt))  # inspect formatting; no model inference
m = scope.measure(prompt=prompt)
m.credences         # {"positive": 0.91, "negative": 0.03, "neutral": 0.06}
m.top_labels        # ("positive",) — all winners, including ties
m.top_label_confidence    # 0.87 — strength of the most likely label (§6.1)
m.entropy_confidence      # 0.67 — concentration of the whole vector (§6.1)
m.margin_confidence       # 0.85 — top label's lead over the runner-up (§6.1)
```

The numbers are illustrative, not benchmark results. The model's native template
opens the answer position; the library adds no answer cue. Each candidate has two
spellings: the candidate itself and one ASCII space followed by it. Whether a
model writes that space depends on what precedes the answer, so both are scored
and a candidate's credence is the total probability of its two spellings (§4.3,
§5). Result keys remain the declared candidates.

The library implements the **generic measurement machinery only**. No earnings-call /
transcript / portfolio / event-study logic lives here — that belongs in the analysis
codebase that consumes `credences`.

## 2. Terminology

| Term | Meaning | Source |
|------|---------|--------|
| **credence** | the model's degree of belief in an answer = its probability for that candidate, in [0,1] | Titelbaum 2022; SEP "Bayesian Epistemology" |
| **credence vector** / **credence distribution** | the normalized distribution over the candidate set — `Measurement.credences`; the primitive object returned by the library | — |
| **top-label confidence** | normalized confidence in the model's most likely label, derived from the credence vector under equal-cost misclassification (§6.1) | relates to CDS 2026 in the binary case |
| **entropy confidence** | normalized reverse entropy of the credence vector; concentration of the whole distribution (§6.1) | extends the reverse-entropy ordering emphasized by CDS 2026 |
| **margin confidence** | the top credence minus the second-highest credence; decisiveness against the nearest competitor (§6.1) | relates to CDS 2026 in the binary case |
| **inner confidence** | the CDS binary measure `|p − 1/2|`; a literature reference point, not the primitive API concept | CDS 2026 (NBER w34965) |
| **declared confidence** | the model's *self-reported* confidence number — biased; a contrast, not what we measure | CDS 2026 |
| **candidate** | an answer value (`"positive"` or `"A"`), trimmed of surrounding whitespace for scoring; it remains the result key as supplied | — |
| **spelling** | one of a candidate's two scored texts: the candidate as written (*bare*) or after one space (*spaced*) | — |
| **coverage** | a diagnostic: the model's unrestricted probability that its first answer token begins one of the allowed spellings (§6.5); not a measure of accuracy | — |
| **label geometry** | researcher-supplied structure on labels — e.g. sentiment scores `(-1, 0, 1)` — used for application-specific projections, not part of the generic credence measurand | — |

We use "credence" rather than "confidence" because "confidence" is overloaded (CDS
itself splits it into *inner* vs *declared*) and because the package returns the
whole distribution. Confidence summaries are measures computed from that
distribution.

## 3. Scope and non-goals

**In scope (v1):**

- One prepared object — a **`Credoscope`** bound to a model + candidate set;
  `.measure(...) → Measurement | list[Measurement]` for scalar or sequence input
  (credences + summaries + audit trail).
- Multi-token candidates via a token trie with exactly-specified probability
  semantics (§5, §7).
- Two spellings per candidate, bare and prefixed with one ASCII space, each in
  its canonical tokenization; a candidate's credence sums both (§4.3, §5, §8).
- Native assistant opening by default; optional caller-owned assistant prefill
  through `messages=` (§4.2). No package-supplied answer cue or template option.
- A no-inference `.preview()` of the actual compiled context, every scored token
  path, and the readout plan, for checking formatting before measurement (§4.4).
- **Reasoning bypass** for chain-of-thought models (§9). Generated reasoning and
  comparisons between reasoning modes are deferred.
- One hardened backend: **MLX** (Apple Silicon), with arbitrary-token logit
  access at supplied contexts. This capability, not physical locality, defines
  the backend contract (§10).
- Property-based + tokenizer-only + opt-in real-model tests (§14).

**Explicit non-goals (v1):**

- ❌ A family of task verbs (`classify`/`rate`/`label`). One core operation.
- ❌ Wide numeric rating scales (0–100) as a first-class interface.
- ❌ Multi-attribute batched rating with conditional column schemas.
- ❌ Task-prompt construction, option-order randomization, code assignment, or semantic
  remapping. These belong to the caller; a possible helper is described in
  `POSSIBLE_FUTURE_EXTENSIONS.md`. Task wording, assistant prefills, and semantic
  mappings remain caller-owned; the library defines the two-spelling policy (§4.3).
- ❌ Restricted top-K probability APIs, partial readouts, or imputed token scores.
- ❌ Whitespace variants beyond the one optional initial ASCII space, case or
  synonym expansion, non-canonical tokenizations of a spelling, or mandatory JSON.
- ❌ Extracting probabilities from *inside* a chain-of-thought.
- ❌ Distributed scheduling, checkpoint/resume, `save_dir` plumbing.
- ❌ DataFrames as a hard dependency (optional adapter only, §12).
- ❌ KV-cache operations in the backend interface; multi-level cascading caches (§13).
- ❌ Many backends at once. V1 = MLX; an NVIDIA backend is deferred (§10.3).

## 4. The core: a prepared Credoscope

A **Credoscope** is an instrument bound to a model + a candidate set, **built once and
reused** across many passages. Candidate validation is shared; whether token
sequences can also be reused depends on the fixed-context continuation contract (§8).

```python
from credences import load_model, build_credoscope

model = load_model("mlx-community/Qwen3-4B-4bit")

scope = build_credoscope(
    model,                  # a loaded credences.Model (§10.1)
    candidates,             # Sequence[str] of exact answer strings — §4.1
    reasoning="bypass",     # explicit chain-of-thought policy; only v1 mode — §9
    end_tokens=None,        # override the end-of-answer token set; only consulted when
                            #   one accepted path is a token-prefix of another — §5.3
)

m  = scope.measure(prompt="...")           # one complete prompt -> Measurement
ms = scope.measure(prompt=["...", "..."])  # sequence -> list[Measurement], 1:1, [] -> []
```

Options after `candidates` are keyword-only. Construction validates configuration;
each measurement encodes both spellings of every candidate together with the
prompt text, then scores all of them in one trie (§8).

There is deliberately **no `temperature` parameter**. Extraction reads the model's
next-token distribution; extraction does not sample, and rescaling logits by a
temperature would change the measurand. Backend numerical variation can still
affect repeated measurements. Only the optional, caller-invoked tie-selection
helper (§6.4) uses randomness; it never changes a measurement.

A Credoscope's configuration is immutable after build and reusable across a corpus;
measurement calls are independent and have no cross-call semantic state. Reuse of
an immutable token-trie plan cannot affect results (§8.3); model/KV state is not
carried between measurements (§13). There is
no `__call__` — `.measure()` is the one measurement entry point; `.preview()` is
its inspection-only companion (§4.4). `Credoscope` is exported for type
annotations; `build_credoscope` is the blessed constructor.

### 4.1 `candidates` — answer strings (bound at build)

`candidates` is a `Sequence[str]`, for example
`["positive", "negative", "neutral"]` or `["A", "B", "C"]`. Each string is
an answer value and, exactly as supplied, the corresponding key in
`Measurement.credences`. Each candidate is scored in two spellings: the
candidate and `" " + candidate` (§4.3). The Credoscope has no separate knowledge
of semantic labels, descriptions, codes, or option presentation in the task prompt.

**Trimming.** Leading and trailing whitespace (as removed by Python's
`str.strip()`) is removed from each candidate before scoring; internal whitespace
is kept exactly. Whether an answer begins with a space is the model's choice,
which the two spellings already measure: a candidate written as `" yes"` would
turn the spaced spelling into a double space and leave no unspaced option.
Trailing whitespace would make the model write a space before stopping. Internal
whitespace, as in `"very positive"`, is part of what the answer says, so it is
not collapsed. The documentation states this rule; the result keys stay as the
caller wrote them so lookups keep working.

After trimming, candidates must be non-empty and unique; `["yes", " yes"]` is
rejected because both would score the same text. Sequence order is the canonical
candidate order used to present results, including all tied winners (§6.4),
without selecting one winner. A bare string and all mapping types are rejected;
candidate descriptions and semantic metadata are not accepted here.
`K = len(candidates)` must be at least 2. A one-option forced choice always returns
credence 1.0 and therefore contains no information; `build_credoscope` rejects it.

### 4.2 Prompt inputs and chat specs

`.measure()` takes **exactly one** of two keyword-only inputs. Each input accepts
either one value or a `Sequence` of values. A scalar returns `Measurement`; a
sequence returns `list[Measurement]` in input order:

```python
scope.measure(prompt="...")             # str: one fully-written user message
scope.measure(prompt=["...", "..."])   # Sequence[str]
scope.measure(messages=[...])           # ChatSpec: an explicit message list
scope.measure(messages=[[...], [...]])  # Sequence[ChatSpec]
```

For `prompt=`, `str` is always scalar even though Python strings are sequences.
For `messages=`, a non-empty sequence of message mappings is one
`ChatSpec`, while a sequence of `ChatSpec` objects is multiple inputs. Because an
empty chat is not valid, `messages=[]` unambiguously means zero inputs and returns
`[]`. Public overloads expose the corresponding scalar and sequence return types
to type checkers; runtime dispatch follows the same rules.

- **`prompt=`** — a string, treated as the complete content of a single user
  message. The Credoscope applies the model's chat template around it. On a chat
  model this is exactly equivalent to `messages=[{"role": "user", "content": prompt}]`;
  a cue at the end of the string is still user content, not an assistant prefill.
- **`messages=`** — a `ChatSpec`: a list of `{"role": "system"|"user"|"assistant",
  "content": str}` dicts. Completed assistant turns are allowed as few-shot
  examples. If the final message is a user message, open a new assistant turn.
  If it is an assistant message, treat its content as an explicit final-answer
  prefill and continue that turn without closing it or adding another assistant
  header. A final system message is not a measurement position and raises
  `ValueError`. Other role ordering follows the model's template requirements.

```python
messages = [
    {"role": "user", "content": "Classify the sentiment of this passage..."},
    {"role": "assistant", "content": "Answer:"},
]
print(scope.preview(messages=messages))
m = scope.measure(messages=messages)
```

The prefill is supplied text, not generated reasoning. Keep it exactly as
written, including trailing spaces and newlines, at the final answer position
after the model-specific reasoning bypass. Some chat templates (Llama 3, Gemma,
Mistral) trim message content, and `transformers` then drops the prefill's
trailing whitespace without warning; credences puts that whitespace back after
rendering (§8.1). If a template changes the prefill in any other way, raise
before scoring rather than relocate, rewrite, or drop it. A trailing space needs
no special care from the caller: it merges into the answer tokens (§8.2).

`messages=` requires a chat template; on a template-less base model it raises
`ValueError` naming the model. Only `reasoning="bypass"` is supported in v1;
generated reasoning and post-trace prefills remain deferred (§9).

Keeping the answer options in `prompt=` or `messages=` consistent with the bound
candidates is the caller's responsibility. Extraction scores the candidate strings
unconditionally and cannot verify their meaning or presentation in the prompt.
When codes stand for semantic labels, the caller must retain the per-input mapping
and join it to the returned candidate credences. A future standalone rendering
helper is described in `POSSIBLE_FUTURE_EXTENSIONS.md`.

Raw completion mode: on a template-less base model, `prompt=` is the complete
prompt text, with no chat template, added cue, or implicit newline. The caller
includes any desired prefill in that string. The same two spellings and
in-context encoding rules apply (§8.2). On a chat model, use `messages=` for an
assistant prefill rather than embedding chat markers in `prompt=`.

### 4.3 Two spellings per candidate

For each (trimmed) candidate `c`, v1 scores exactly two spellings:

```text
bare:   c
spaced: " " + c
```

A candidate's credence is the **total** probability of its two spellings (§5.1).

**Why both.** In written text one space separates words, and most tokenizers
fold that space into the following word's token (`" yes"` is a single token,
different from `"yes"`). Whether the model writes the space depends on what
precedes the answer, and differs between models:

| Text before the answer | Spelling the model's chat format suggests (inferred from templates, not measured) |
|---|---|
| Chat header ending in a newline (Qwen, Llama 3, Gemma, gpt-oss) | bare: `"yes"` |
| Mistral's `[/INST]` (its template puts a space before replies) | spaced: `" yes"` |
| A prefill such as `"Answer:"` | spaced: `" yes"` |

Rather than guess per model, credences scores both and adds them. When one
spelling dominates, the total is close to that spelling alone; when the model is
undecided about the space, both contribute, and the credence changes smoothly
with the model's scores instead of jumping between spellings. In the five-label
example of §5.2, adding instead of choosing the stronger spelling moves no
credence by more than 0.0014.

**Why only these two.** Anything else in front of the answer (a newline, a
second space, `**`, a quotation mark) is the model choosing to format its reply,
like starting with "The"; those outputs are not answers in the candidate's form
and are excluded by the constraint. Capitalization and synonyms change the word
itself, so they are the caller's choice of candidates, not an automatic
variant. The added character is one ASCII space (`U+0020`), only before the
whole candidate, never between its internal tokens. There is no formatting
argument, placeholder syntax, or automatic answer cue. Prefills belong to the
input (§4.2).

**One tokenization per spelling.** Each spelling is scored in its canonical
tokenization: the one the tokenizer produces when the spelling is encoded
together with the prompt text (§8.2). Other token sequences that spell the same
text, such as `["y", "e", "s"]`, are not added. Tokenizers never produce them,
models rarely generate them, and Geh et al. (2024) find that the canonical
tokenization usually carries almost all of a text's probability. For `Tokens`
under Llama 2 it carries about 99.6%, with 51 alternatives sharing the rest.
Their benchmark gains from adding non-canonical tokenizations required tuning a
mixing weight on held-out data, which a measurement should not depend on.

Illustrative splits, after a context that ends in a newline:

| Candidate | Bare path | Spaced path |
|---|---|---|
| `"positive"` | `["positive"]` | `[" positive"]` |
| `"very positive"` | `["very", " positive"]` | `[" very", " positive"]` |
| `"elequent"` | `["ele", "quent"]` | `[" ele", "quent"]` |
| `"eloquent"` | `["elo", "quent"]` | `[" eloquent"]` |
| `"1"` | `["1"]` | `[" ", "1"]` (digits do not absorb a space in Llama 3 or Qwen) |

These splits are illustrative, not promises about a vocabulary. The leading
space can change more than the first token, so each spelling is encoded whole
rather than by swapping its first token id. Some tokenizers give both spellings
the same token path in a given context (Phi-3.5 after its assistant header);
such a path is scored and counted once.

The policy is a bounded measurement constraint, not a claim to capture every
way the model could express a label or to eliminate prompt sensitivity. Use
`.preview()` to inspect the context and the exact paths that will be scored.

### 4.4 Preview the actual model input

`scope.preview(...)` makes prompt formatting inspectable without running the
model. It accepts exactly the same `prompt=` or `messages=` inputs and dispatch
rules as `.measure()` (§4.2). One input returns a human-readable `str`; a sequence
returns an aligned `list[str]`, with an empty sequence returning `[]`.

```python
prompt = 'Rate the sentiment as "positive", "negative", or "neutral". Answer:'
print(scope.preview(prompt=prompt))
# The cue stays in the user message; no assistant cue is inserted automatically.

for report in scope.preview(prompt=["First passage...", "Second passage..."]):
    print(report)
```

Each report includes:

- The full rendered prompt text, without truncation or hidden special tokens.
  This includes actual role delimiters, bypass text if present, and any supplied
  assistant prefill. Show both readable text and an escaped representation so
  trailing spaces and newlines can be inspected.
- Where scoring starts: the scoring position `R` (§8.2). When the prompt's last
  characters merge into the answer tokens, such as a trailing space, show that
  text and say it is scored as the start of each answer.
- Every candidate in declared order, with its trimmed text and both spellings
  labeled `bare` and `spaced`: their escaped text and complete token-id paths
  from `R`. Mark a path shared by both spellings.
- The resolved `END` token ids when an end-vs-continue decision exists (§5.3).
- The readout plan: the scoring position and every later prefix at which scores
  will be requested, with the token ids requested there (§7.2).

Label prompt and answers separately. Candidates are supplied alternatives,
not generated answers or predictions. The report is for inspection, not a second
prompt to feed back into `.measure()` or a machine-readable serialization format.

**One preparation path:** preview and measurement must share input validation,
native rendering, in-context encoding, path and `END` validation, and plan
construction. There is no simplified preview renderer. Because the scored paths
do not depend on the model's scores, preview shows exactly the prompt, paths, and
readouts a measurement will use; only the logits and credences need inference.
Invalid inputs or boundaries raise the same preparation errors; no partial batch
report is returned. Backend/score errors can occur only during measurement.
Preview does not query logits, generate text, allocate model/KV state for a call,
or change subsequent measurements. It uses the already-built scope's tokenizer
and configuration; this does not imply a separate model-loading API.

Return the report without printing, logging, or saving it automatically. Do not
remove duplicate cues, infer prompt intent, or warn based on wording heuristics.
The purpose is to let the researcher inspect their own full inputs, not to add
prompt metadata to every `Measurement`.

## 5. What is measured - the math

This section is normative. Every number the library returns is defined here.

### 5.1 The measurand: constrained distribution over both spellings

The credence of a candidate is the probability that a decoder, allowed to write
only the candidates' spellings, writes one of that candidate's spellings.

Let `S(c)` be the distinct token paths of candidate `c`'s two spellings (one path
when both spellings tokenize identically), all continuing from the shared
scoring position `R` (§8.2). Build one trie containing every path of every
candidate. At prefix `P`, the allowed set `A(P)` contains the distinct next
tokens of paths through `P`, plus the model's `END` tokens when a path of one
candidate ends at `P` and a path of another candidate continues (§5.3). At a
leaf, stop without an additional model query.

For a nonempty allowed set of size `n = |A(P)|`, with raw model scores `z_P`,
define the transition rule:

    p_tilde(t | P) = 1                              if n = 1
                    1/n                            if all allowed scores are -inf
                    exp(z_P(t) - logsumexp(z_P(A))) otherwise

A singleton is forced, regardless of its model score. At a branching node with
at least one finite score, individual `-inf` scores get probability zero and the
finite scores divide the total mass. NaN, positive infinity, and missing
requested scores are errors, not inputs to the uniform fallback (§10.2).

The all-`-inf` case is an explicit uniform **token-level fallback**, not an
inference that the model has equal beliefs over labels. A nested trie need not
yield uniform final label credences when its branch transitions are uniform.
Never replace small finite scores with this fallback.

A path's probability is the product along it, and a candidate's credence is the
sum over its paths:

    pi(T)        = product of p_tilde(next token | prefix) along T
                   * p_tilde(END | T)   if another candidate's path continues there
    credence(c)  = sum of pi(T) for T in S(c)

Here `p_tilde(END | P)` is the sum of transition probabilities for the distinct
tokens in `END`. A shared edge contributes once to its node's normalizer, even
when several paths or candidates use it. Every path belongs to exactly one
candidate, so credences sum to 1 by construction, with no label-level
renormalization.

**Interpretation:** a decoder restricted to these paths, with the same forced
transitions and uniform fallback, would sample this distribution, and the
credence of `c` is how often it writes `c` with or without the leading space. It
is not a promise of parity with arbitrary structured-output engines or of
prompt-independent, calibrated belief.

**Why this is not global conditioning:** computing each complete answer's joint
unconstrained probability and renormalizing at the end is a different target.
Local masks redistribute mass at each decision and forced steps contribute 1.
See *Grammar-Aligned Decoding* (References) for the distinction between local
masking and conditioning. Full joint probabilities would also require
full-vocabulary normalizers at every step.

### 5.2 Worked example (five candidates)

Candidates `["very negative", "negative", "neutral", "positive", "very positive"]`.
Assume these illustrative probabilities at the scoring position:

| Starting piece | Bare-token probability | Spaced-token probability |
|---|---|---|
| very | 0.10 | 0.001 |
| negative | 0.65 | 0.002 |
| neutral | 0.20 | 0.001 |
| positive | 0.02 | 0.001 |

All eight distinct first tokens are allowed, so the root normalizer is
`0.10 + 0.001 + 0.65 + 0.002 + 0.20 + 0.001 + 0.02 + 0.001 = 0.975`, with `"very"`
counted once although two candidates start with it. After `"very"`, let
`p(" negative") = 0.95` and `p(" positive") = 0.01` (local denominator 0.96).
After `" very"`, let `p(" negative") = 0.90` and `p(" positive") = 0.05` (0.95).

```text
credence(negative)      = (0.65 + 0.002)/0.975                        = 0.6687
credence(neutral)       = (0.20 + 0.001)/0.975                        = 0.2062
credence(positive)      = (0.02 + 0.001)/0.975                        = 0.0215
credence(very negative) = (0.10/0.975)(0.95/0.96) + (0.001/0.975)(0.90/0.95) = 0.1025
credence(very positive) = (0.10/0.975)(0.01/0.96) + (0.001/0.975)(0.05/0.95) = 0.0011
                                                                 sum = 1.0000
```

There are three readouts: the scoring position (all eight first tokens), the
`"very"` branch, and the `" very"` branch. Choosing only the stronger spelling of
each candidate would give `0.6701, 0.2062, 0.0206, 0.1020, 0.0011`, differing by
at most 0.0014, but would jump if two spellings' first-token scores crossed.

*End-vs-continue example.* For `"increase"` and `"increase substantially"`, the
bare paths `["increase"]` and `["increase", " substantially"]` share a prefix,
and so do the spaced paths `[" increase"]` and `[" increase", " substantially"]`.
Both prefixes need `END union {" substantially"}` readouts.

### 5.3 The end-of-answer set (`END`)

An end-vs-continue decision is needed when a path of one candidate is a strict
prefix of a path of a **different** candidate. Within one candidate, one
spelling's path cannot be a prefix of the other's (the texts `c` and `" " + c`
never prefix each other once `c` is trimmed); if a tokenizer quirk produced one,
raise before scoring (§7.1, §8.2).

Resolution order: explicit `end_tokens=` override on `build_credoscope` ->
blessed-model registry (§9.2) -> for unlisted models, `{tokenizer.eos_token_id}`
plus additional special tokens matching a known end-of-turn list, with a warning.
Resolve and deduplicate `END` at build time.

Every path is known before inference (§8.3), so preparation checks every
end-vs-continue boundary: if a boundary lacks a usable `END`, or its continuation
token is also in `END`, raise `ValueError`. Preview performs the same check.

Normalize over individual token ids, including all distinct `END` ids, then sum
the `END` probabilities. If all scores are `-inf`, each allowed token gets
`1 / |A(P)|`: with two end tokens and one continuation token, stopping gets
`2/3`, not `1/2`. Uniform fallback is over tokens, not over grouped outcomes.

**Do not implement:** crediting the shorter candidate with
`1 - sum p(candidate continuations)` over the full vocabulary. That would count
unrelated words and punctuation as choosing to stop. Only actual `END` tokens
represent stopping at an end-vs-continue branch.

### 5.4 Failure modes these definitions exclude

1. **Choosing one spelling.** Keeping only the higher-scoring spelling of each
   candidate is a different, discontinuous measurand. Score every path.
2. **Per-label or per-spelling root normalization.** If two paths begin with
   `"pos"`, that token gets one root weight; divide it at a later branch. Counting
   it twice, or normalizing each candidate's spellings separately, changes the
   result.
3. **Double-counting one path.** When both spellings of a candidate tokenize to
   the same path, it is one outcome and is counted once.
4. **Replacing only the first token.** A spaced `" positive"` token must not
   inherit `"itive"` from the bare `["pos", "itive"]` path; each spelling keeps
   its own complete tokenization.
5. **Encoding answers on their own.** A spelling encoded without its prompt can
   get the wrong tokens (Mistral encodes a lone `"yes"` as `"▁yes"`, its spaced
   token). Encode in context (§8.2).
6. **Mixed probability conventions.** All decisions use §5.1's rule, including
   end-vs-continue nodes. No unnormalized terminal factors or complement mass.
7. **Zero-score arithmetic.** Do not evaluate `-inf - (-inf)`, replace tiny
   finite scores by zero, or use uniform fallback when any allowed score is finite.

The §14 parity test compares trie scoring with an independent reference that
enumerates every path, applies the chain rule including forced transitions and
the all-`-inf` fallback, and sums each candidate's paths.

### 5.5 Numerical rules

- Compute in **natural-log space**. With at least one finite allowed score,
  step logprob is `score(t) - logsumexp(allowed scores)`. A forced step contributes
  0; an all-`-inf` branching step contributes `-ln(n)` for each of its `n`
  tokens. Implement these cases explicitly. An `END` event uses `logsumexp`
  over its normalized token logprobs, including in the fallback case.
  Bound positive grouped-event roundoff at log probability 0 only when excess
  mass is within relative tolerance `1e-12`; larger excess mass raises.
  Preserve negative event logs, including tiny finite values and `-inf`.
- A path's log probability is the sum of step logprobs along it. A candidate's
  log credence combines its paths with the same grouped-event sum, because its
  spellings are mutually exclusive outcomes. For `"very negative"` in §5.2:

      log credence = logsumexp(ln(0.10) - ln(0.975) + ln(0.95) - ln(0.96),
                               ln(0.001) - ln(0.975) + ln(0.90) - ln(0.95))

  Exponentiating gives about `0.1025`. Preserve a finite log credence even when
  exponentiation underflows to `0.0`; exact zero mass has log credence `-inf`.
  Determine winning labels from unrounded log credences (§6.4).
- The backend casts requested bf16/fp16 logits to **float32** before returning
  them. Pure allowed-set log-softmax, grouped-event sums, and path accumulation
  use Python's higher-precision floats (normally float64). Subtract the maximum
  before normalization so large common offsets cannot erase branch weights.
  This also preserves finite log differences across the finite float32 range.
  Avoid forming full-vocabulary probabilities before extracting allowed scores.
  Higher-precision arithmetic cannot recover information lost in the model's
  logits; quantized weights still measure the loaded quantized model.
- A fixed absolute probability error can be large relative to a small probability.
  Log-space arithmetic avoids additional underflow; it cannot undo upstream
  quantization/logit error.
- Entropy uses `0 * log 0 = 0`.
- Measurement has no sampling or cross-call semantic state. Bitwise reproducibility
  is not promised across kernels, batching, devices, or environments. Researchers
  own experiment tracking.
- **Token-granularity caveat:** the root compares tokens that cover different
  amounts of text, such as `"pos"` and `" positive"`, and a standalone space token
  (the spaced spelling of a digit) can carry broad continuation mass. This is
  part of the specified token-level procedure, not semantic equivalence or
  calibration. Inspect the paths with preview and in `raw.scored_paths`.

## 6. Result objects

```python
@dataclass(frozen=True)
class ScoredPath:
    spellings: tuple[str, ...]             # ("bare",), ("spaced",), or both when the two
                                           # spellings share one token path
    token_ids: tuple[int, ...]             # the path's tokens after the scoring position
    logprob: float                         # natural-log probability of this path (§5.1)

@dataclass(frozen=True)
class RawReadout:
    credence_logprobs: dict[str, float]    # per candidate: ln(credence) — natural log; the
                                           # log-space source of Measurement.credences
    scored_paths: dict[str, tuple[ScoredPath, ...]]
                                           # per candidate: every scored path, bare first;
                                           # their probabilities sum to the credence
    context_token_count: int                # tokens before the scoring position R
    coverage: float                         # diagnostic, not accuracy (§6.5)

@dataclass(frozen=True)
class Measurement:
    credences: dict[str, float]   # the forced-choice distribution (§5.1), keyed by candidate
    entropy: float                # Shannon entropy of credences, in bits
    top_labels: tuple[str, ...]   # all exact maximizers of log credence; candidate order (§6.4)
    top_label_confidence: float   # normalized strength of the most likely label (§6.1)
    entropy_confidence: float     # 1 - entropy / log2(K) (§6.1)
    margin_confidence: float      # p(top1) - p(top2) (§6.1)
    raw: RawReadout               # always populated, never None

    def to_dict(self) -> dict: ...   # stable, documented JSON-safe schema:
                                     # tuples become lists; log -inf becomes null;
                                     # finite logs remain finite even if exp(log) underflows.
```

### 6.1 Summaries and their caveats

- **`top_label_confidence`** answers: *how strongly does the model favor its most
  likely label?* For `K > 1`,

      top_label_confidence = (p_(1) − 1/K) / (1 − 1/K)

  where `p_(1)` is the largest credence. In plain terms, if we must reduce the
  credence vector to one categorical label and treat every misclassification as
  equally costly, the model-implied probability of error is `1 − p_(1)`.
  `top_label_confidence` rescales the reduction in that error probability so that
  0 means no improvement over uniform guessing and 1 means all credence is on one
  label. In decision-theoretic terms, this is reverse normalized Bayes risk under
  zero-one loss. For `K = 2`, it equals `2 · |p − 1/2|`: exactly twice the CDS
  binary inner-confidence measure, preserving its ordering. Divide by 2 to report
  on the CDS scale.
- **`entropy_confidence`** answers: *how concentrated is the whole credence vector?*
  For `K > 1`,

      entropy_confidence = 1 − entropy / log₂(K)

  where `entropy` is Shannon entropy in bits. Base 2 gives a standard information
  unit and preserves entropy's dependence on the number of alternatives;
  `entropy_confidence` already supplies the normalized, base-independent summary.
  This summary uses all coordinates of the credence vector, not just the winner,
  and exactly preserves reverse Shannon
  entropy ordering for fixed `K`. In the binary case it is a monotone transform of
  the CDS inner-confidence statistic; for `K > 2`, it deliberately answers a
  different question than top-label confidence because entropy also depends on
  how the nonwinning mass is distributed.
- **`margin_confidence`** answers: *how decisively does the preferred label beat
  its closest competitor?* It is `p_(1) − p_(2)`. For binary tasks it also equals
  `2 · |p − 1/2|`, hence twice the CDS
  measure. For `K > 2`, it is useful when the practical concern is the nearest
  rival rather than the entire tail of alternatives.
- These three summaries are **derived from the credence vector from first
  principles**. CDS is the key literature bridge because its binary inner
  confidence is the shared special case or monotone reference point, but the API
  is not organized around reproducing one scalar from that paper.
- The pure result builder checks that exponentiated log credences sum to 1
  within relative tolerance `1e-12`, without label-level renormalization.
  Confidence summaries are bounded to `[0, 1]` to remove boundary roundoff;
  credences and entropy are not rounded or renormalized.
- Experiment metadata, model identifiers, versions, and semantic mappings live
  in caller-owned data, not a per-measurement metadata object. There is no
  `Measurement.reasoning`: bypass is the only supported mode. The raw readout
  contains measurement-specific log credences, scored paths, and coverage, not
  run metadata.

No DataFrames in the core. An optional `credences.frames` adapter (separate import,
optional `pandas`/`polars` extra) converts `list[Measurement]` to a DataFrame.

### 6.2 Application-specific projections from credences

The generic package summaries above require no structure beyond a probability
vector over candidates. Application-specific projections operate on semantic
labels, after any caller-owned code mapping has been applied. Ordered sentiment is
the canonical example: once a researcher assigns scores such as

    negative = -1, neutral = 0, positive = 1

the same credence vector can be projected into an expected sentiment score:

    expected_score = Σ_i p_i x_i

and an ordinal dispersion:

    score_sd = sqrt(Σ_i p_i (x_i − expected_score)^2)

Those quantities are not generic confidence measures. They combine model credences
with a researcher-supplied **label geometry**. That is often exactly the right move
for a domain application, but the extra structure should be explicit.

For sentiment, this distinction is the point. Taking the argmax of the final
credence vector discards information:

    A = (negative=0.01, neutral=0.98, positive=0.01)
    B = (negative=0.45, neutral=0.54, positive=0.01)

Both have `neutral` as their highest-credence label. This is not a claim about
token-by-token greedy generation, which need not select the most probable
complete candidate. The vectors say different things:
`A` is overwhelmingly neutral, while `B` is barely neutral and leans strongly
negative. Under the `(-1, 0, 1)` geometry, their expected sentiment scores are
approximately `0.00` and `-0.44`.

A downstream finance regression can therefore compare three levels of information:
the discrete argmax label, a one-dimensional expected-sentiment projection, and the
full credence vector. With three labels, regressing returns on expected sentiment
`S = p_positive − p_negative` imposes the restriction that positive and negative
credences have equal and opposite effects. Regressing on the credence vector
directly (omitting one probability when using an intercept) lets the data test that
restriction. This is application logic, not core package behavior, but it is the
main reason the vector must remain the primary object.

### 6.3 Sequence inputs

For any accepted input kind, passing a sequence is semantically equivalent to
calling `.measure()` once per element and collecting the results in order. An
empty sequence returns `[]`. V1 executes the sequence serially and is fail-fast:
an exception for element *i* propagates without returning partial results. The
sequence surface exists so backends can later add transparent batching without an
API change. Progress reporting, streaming persistence, skip-and-continue behavior,
and distributed retries belong in the caller's loop.

### 6.4 Ties and categorical summaries

`top_labels` contains every candidate whose unrounded log credence equals the
maximum. Equality is exact in the computed values: no implicit tolerance and no
random tie-breaking. Canonical candidate order makes presentation stable, not a
recommended way to select a winner. `len(m.top_labels) > 1` identifies ties; a
zero rounded margin is not a reliable substitute for that test.

Two optional pure helpers support downstream counting:

- `choose_top_label(m, *, rng) -> str`: uniformly select one member of
  `m.top_labels` using a caller-supplied `random.Random`-compatible generator.
  No global RNG or hidden seed. It does not modify the measurement.
- `top_label_weights(m) -> dict[str, float]`: return only the tied winners,
  each with weight `1 / len(m.top_labels)`. Summing these weights across inputs
  gives fractional winner counts. These are counting weights, not credences.

For `top_labels == ("A", "B")`, the weights are `{"A": 0.5, "B": 0.5}`,
even if the actual credences are `{"A": 0.4, "B": 0.4, "C": 0.2}`. No warning
is emitted for a tie; it is a valid result exposed directly in the return type.

### 6.5 Coverage: a diagnostic, not a quality score

`raw.coverage` is the model's unrestricted probability, at the scoring position,
that its next token begins one of the allowed spellings: the full-vocabulary
softmax summed over the allowed first tokens. It is an upper bound on how likely
the model was to answer with a candidate unprompted, because a token that begins
a spelling may continue differently.

Credences do not depend on coverage, and low coverage is **not** evidence of
worse credences. Consider two models answering a yes/no question:

| Model | Next-token probabilities | Coverage | Credences |
|---|---|---|---|
| 1 | `"The"` 0.90, `"yes"` 0.07, `"no"` 0.02 | 0.09 | yes 0.78, no 0.22 |
| 2 | `"yes"` 0.70, `"no"` 0.20, `"The"` 0.01 | 0.90 | yes 0.78, no 0.22 |

Model 1 habitually opens with "The"; model 2 answers directly. Their credences
are identical, and either could be the more accurate in a given setting; whether
coverage relates to accuracy is an empirical, model-dependent question. Coverage
is reported for inspection and research. The library never filters, reweights,
or warns on it.

The scoring position is always read, even when only one token is allowed there,
so coverage is always defined. It needs one extra number from the backend: the
log normalizer over the whole vocabulary at that position (§10.2).

## 7. The token trie

The trie is pure and backend-independent: token paths in, a readout plan out,
credences back. The Credoscope owns the model calls.

### 7.1 Inputs and structure

Input: `paths: dict[str, tuple[tuple[int, ...], ...]]`, the distinct paths of
each candidate's spellings (one or two per candidate), all continuing from the
same scoring position `R`.

Each node is a unique prefix `P`, with distinct child token ids and at most one
candidate owning a path that ends there. Merge shared prefixes; never duplicate
an edge's weight because several paths follow it. A node where one candidate's
path ends and another candidate's path continues is an end-vs-continue node,
whose allowed set includes `END`.

Validation:

- Empty paths, and identical paths owned by different candidates, raise
  `ValueError`. A candidate's own two spellings sharing one path is allowed; it is
  stored once.
- A path that is a strict prefix of another path of the same candidate raises
  (§5.3).
- At an end-vs-continue node, `END` must be nonempty and disjoint from child ids.

### 7.2 The readout plan

The plan depends only on the paths and `END`, never on scores, so it is built
before any model call and preview can show it.

- Always read the scoring position `R`, requesting every allowed first token and
  the full-vocabulary log normalizer used for coverage (§6.5).
- For each non-root node `P` with at least two allowed token ids, query
  `next_token_logits(R + P, A(P))`.
- Skip non-root singleton nodes: their transition is forced, with probability 1,
  even if the unqueried score would be `-inf`. A singleton root is still read,
  for coverage only; its transition is still forced. Leaves stop without
  querying an end score.

Forced prefixes must still be appended before scoring a later branch; skipping
a readout does not delete tokens from model history. With `["very positive",
"very negative"]`, the plan reads the scoring position, then the `"very"` and
`" very"` branches. If every first token belongs to one candidate's spelling,
the scoring position is the only readout. Forward-pass cost remains a backend
concern; readout count alone is not a wall-clock guarantee.

### 7.3 Scoring

Given the paths and validated requested-token logits:

```text
for each node P whose allowed set A(P) is nonempty:
    if len(A(P)) == 1:
        step_logprob(P, sole token) = 0       # forced
    elif all scores over A(P) are -inf:
        step_logprob(P, each token) = -ln(len(A(P)))
    else:
        logdenom = logsumexp(scores over A(P))
        step_logprob(P, token) = score[P][token] - logdenom

for each path T:
    path_logprob(T) = sum of step logprobs along T
    if another candidate's path continues through T:
        path_logprob(T) += logsumexp(step_logprob(T, e) for e in END)
        # bound positive grouped-event roundoff at 0 per §5.5

for each candidate c:
    log_credence(c) = logsumexp(path_logprob(T) for T in S(c))
        # same grouped-event bound
```

The normalized transitions and stopping events partition mass among paths, and
each path belongs to one candidate, so final credences sum to 1 without another
normalization. A path with a zero-probability edge retains log probability
`-inf`; uniform transitions farther down that path do not restore its lost
incoming mass.

**Parity invariant (§14):** trie scoring must match an independent reference
that enumerates every path, applies the naïve chain rule, and sums each
candidate's paths. Include shared roots, shared paths of one candidate,
singleton forcing, mixed finite/`-inf` scores, and uniform fallback. A reference
that keeps only one spelling per candidate or normalizes per label is incorrect.

### 7.4 What the trie is not

It is not a guided-generation engine, KV cache, tokenizer, or model client. It
does not choose between spellings; it scores all of them.

## 8. In-context tokenization

Each spelling is tokenized as part of the full text the model would see, the
prompt followed by the answer, and scored from the tokens that text shares with
the prompt. The prompt's text is never changed; only the position where scoring
starts can move.

### 8.1 Render the prompt text

For a chat model, render completed messages with the native template. If the
last input message is from the user, open the assistant's final answer position,
including the model-specific reasoning bypass (§9). If the last message is from
the assistant, its content is a final-answer prefill: render the same opening
and bypass, then that exact content, leaving the turn open. An empty final
assistant prefill is equivalent to opening the answer position without a prefill.

Use a supported continuation-of-final-message mechanism or an equivalent tested
model-profile renderer. Do not both open another assistant turn and continue an
existing one. The rendered text must end with the prefill exactly:

- If the template trimmed the prefill's trailing whitespace (Llama 3, Gemma and
  Mistral templates apply `| trim`, and `transformers` then strips the rendered
  text to match), append the removed whitespace.
- If the template changed the prefill in any other way, including removing
  leading whitespace, raise `TokenBoundaryError` naming the model.

Native assistant formatting is model-specific. Preserve role/channel delimiters
and native opening whitespace exactly once. A generation opening need not be
identical to the template's prefix for nonempty assistant content; the profile
must validate both the no-prefill and explicit-prefill cases. Do not invent an
answer cue, move user text into the assistant turn, close the final assistant
turn, or append a second bypass block.

For a template-less base model, the caller's `prompt` is the prompt text. No
model call or per-call KV state is needed to prepare it.

### 8.2 In-context encoding and the scoring position

Tokenizers encode text differently depending on its neighbors. Measured on
real tokenizers:

- **Byte-level tokenizers** (Qwen, gpt-oss, Llama 3, SmolLM) attach a space to
  the following word, so `"Answer: yes"` is `[":", " yes"]`. A prompt ending in
  `"Answer: "` would leave a lone `" "` token before `"yes"`, a sequence the model
  almost never saw in training.
- **SentencePiece tokenizers** add a "dummy" space when encoding text on its own:
  Mistral encodes a lone `"yes"` as `"▁yes"`, its spaced token, while in context
  `"Answer:yes"` ends in a different, unspaced token. Phi-3.5 encodes a lone
  `" yes"` with a stray extra space token.
- **Newlines** can re-split: SmolLM2 encodes `"Answer:\n\n"` with one `"\n\n"`
  token but `"Answer:\n\nyes"` with two `"\n"` tokens.

So each spelling is encoded together with the prompt, and scoring starts where
the encodings stop agreeing:

```text
text      = render_prompt_text(input, reasoning="bypass")       # §8.1
prompt    = encode(text)
for candidate c, for spelling s in (c, " " + c):
    full[c, s] = encode(text + s)
    require decode(full[c, s]) == text + s                     # exact round trip
require decode(prompt) == text
R         = longest common prefix of prompt and every full[c, s]
path[c, s] = full[c, s][len(R):]                               # nonempty
```

`R` is the scoring position, shared by every candidate and spelling. Text after
`R`, such as a prompt's trailing space, becomes the start of every path. The
model then sees exactly the tokens it would see if the full text `"…Answer: yes"`
appeared in its input. This is the "prompt boundary" problem that token healing
addresses (Lundberg & Ribeiro 2023), and it generalizes lm-evaluation-harness,
which moves a context's trailing whitespace into the continuation and encodes
the whole string. Computing the shared prefix, rather than moving whitespace by
rule, also catches merges that are not whitespace and verifies the split instead
of assuming it.

A shared position matters when candidates merge differently. With the prompt
`"…Answer: "` and candidates `"yes"` and `"1"`, Qwen, Llama 3 and gpt-oss merge
the space into `" yes"` but keep it separate from the digit, so `R` ends after
`":"` and the paths are `[" yes"]` and `[" ", "1"]`.

Special tokens: chat templates already write the beginning-of-sequence token
where the model expects it, so encode rendered chat text with
`add_special_tokens=False`. Encoding it with the Hugging Face default doubles that
token on Llama 3, Gemma and Mistral. For a base model's raw prompt, use the
tokenizer's own start-of-sequence policy, and raise if that policy appends
special tokens at the end.

Decoding retains special tokens and disables whitespace cleanup. If any round
trip fails, a path is empty, or one candidate's path prefixes its own other path,
raise `TokenBoundaryError` naming the model and the text. Phi-3.5, whose
tokenizer changes text next to special tokens, fails this check and is
unsupported. Do not rewrite the prompt text, drop a spelling, or change a
segmentation to pass the check.

Encoding requires no inference. There is one canonical path per spelling, not a
search over tokenizations.

### 8.3 Build, measure, and reuse

`build_credoscope` binds model, trimmed candidates, bypass policy, and resolved
`END`. It validates candidate values without requiring a sample prompt.

For each measurement:

1. Render the prompt text and encode every spelling in context (§8.1, §8.2).
2. Validate paths and `END` boundaries (§5.3), then build or reuse the trie plan.
3. Read the scoring position, then each branch node in the plan (§7.2).
4. Score (§7.3) and record every path's probability and the coverage (§6).

Preview stops after step 2 and shows the plan. Scores are neither available nor
needed to inspect the prompt, the paths, or the readouts.

A single-entry immutable plan memo is sufficient. Its key is the ordered
candidate-to-paths mapping plus `END`. Cache only pure plan data, never prompt
text, scores, logits, or model/KV state. Input order, eviction, and preview calls
cannot affect results. Reusing a plan does not skip encoding and validating a
new prompt.

### 8.4 Walkthrough and regression cases

With a byte-level tokenizer and a chat header ending in a newline, the candidate
`"yes"` is scored as follows. Token boundaries are shown with `|`.

| Final assistant prefill | Scored texts | Scoring starts after | Paths |
|---|---|---|---|
| None | `"\nyes"`, `"\n yes"` | the newline | `yes`, ` yes` |
| `"Answer:"` | `"Answer:yes"`, `"Answer: yes"` | `":"` | `yes`, ` yes` |
| `"Answer: "` | `"Answer: yes"`, `"Answer:  yes"` | `":"` (the space merges) | ` yes`, ` | yes` |
| `"Answer:\n\n"` | `"Answer:\n\nyes"`, `"Answer:\n\n yes"` | the newlines (when they do not re-split) | `yes`, ` yes` |

A prefill's whitespace is never removed. A trailing space in a prefill states
that the answer starts with a space, so both scored texts contain it.

**Different segmentations.** Suppose a tokenizer produces:

| Candidate | Bare | Spaced |
|---|---|---|
| positive | `["pos", "itive"]` | `[" positive"]` |
| position | `["pos", "it", "ion"]` | `[" position"]` |

These are illustrative paths, not a claim about a particular vocabulary. The
scoring position allows `"pos"`, `" positive"` and `" position"`. With
`p(" positive") = 0.40`, `p("pos") = 0.30` and `p(" position") = 0.20` (sum 0.90),
and `p("itive") = 0.6`, `p("it") = 0.4` after `"pos"`:

```text
credence(positive) = (0.40 + 0.30 * 0.6) / 0.90 = 0.6444
credence(position) = (0.20 + 0.30 * 0.4) / 0.90 = 0.3556
```

After `"it"`, `"ion"` is forced. Two readouts: the scoring position and `"pos"`.

Regression coverage:

- Scalar chat prompt and equivalent user-only messages compile identically.
  Completed few-shot turns remain completed.
- No prefill, empty prefill, punctuation, trailing-space and trailing-newline
  prefills keep their exact text, including on templates that trim it. Native
  controls and bypass occur exactly once.
- Every spelling round-trips exactly in context, including multibyte text and
  SentencePiece tokenizers. The scoring position moves only when prompt text
  merges into answers, and is shared across candidates that merge differently.
- Different segmentation lengths and suffixes; no first-token-only substitution;
  no non-canonical segmentations.
- Shared first ids are queried once; a path shared by both spellings counts once.
- Mixed finite/`-inf` scores, all-`-inf` uniform token fallback, and forced steps.
  Zero incoming mass stays zero; token-uniform need not mean label-uniform.
- Preview shows the same prompt, paths, and plan that measurement uses, makes no
  backend calls, and does not change later measurements.
- Cold, reused, rebuilt, and preview-interleaved measurements agree within
  backend numerical tolerance.

## 9. Reasoning models: bypass only in v1

`reasoning="bypass"` remains an explicit constructor argument and is the only
supported value. Any other value raises `ValueError` at build time. There is no
per-call override, reasoning generation, or token-cap option in v1.

Bypass disables thinking through the template or supplies an empty/closed thinking
phase before any supplied final-answer prefill. It is a no-op for non-reasoning
models. An assistant prefill is answer text, not a request to continue thinking;
unsupported profile/prefill combinations raise before scoring (§8.1).
There is no per-result reasoning field while bypass is the only mode. Revisit
result fields only when another mode or a generated trace is actually supported.

**Interpretation:** this measures the answer distribution in the bypass context.
It is not a marginal over possible thoughts, and bypass is not claimed to improve
accuracy or calibration. For a model trained to always think, an empty thinking
phase may be off-distribution. Future post-reasoning measurement, stopping rules,
and interactions with post-trace prefills are discussed in
`POSSIBLE_FUTURE_EXTENSIONS.md` §8.

### 9.1 The bypass insert

"Bypass" means: render a final-answer opening without generating a thinking
phase, before any caller-supplied assistant prefill. Any required insert text is
model-specific glue, resolved at `load_model` time in priority order:

1. **Template-native switch** — if the chat template supports it (e.g. Qwen3's
   `enable_thinking=False`), render with that flag. Prefer a supported native
   mechanism without claiming that it preserves answer quality or calibration.
2. **Registry insert** — the blessed-model registry (§9.2) supplies the exact
   insert string, including whitespace glue: e.g. Qwen3/DeepSeek-R1-family
   `"<think>\n\n</think>\n\n"` (an *empty* think block — bare `"</think>"` leaves the
   model wanting a newline next, and the "answer" distribution becomes newline
   mass); gpt-oss/Harmony `"<|channel|>final<|message|>"` (all special tokens — no
   glue problem).
3. **Detected marker + heuristic glue** — template introspection (the template
   mentions `<think>` / `<|channel|>` / `[THINK]` families) with a **loud warning**
   that the model is unblessed and the insert should be verified (one transcript
   eyeball, or the registry sanity prompt).

No marker found ⇒ the model is treated as non-reasoning; bypass is a no-op.
Explicit override: `load_model(..., reasoning_profile=...)`.

### 9.2 The blessed-model registry

A small, shipped, versioned table (`reasoning.py`), consulted at runtime and reused
by tests — **runtime data, not a test fixture**. Per entry: model-id pattern;
exact bypass insert; the native-disable invocation as literal
kwargs (`chat_template_kwargs={"enable_thinking": False}` — a kwargs dict, not a
boolean, so the template-native path is implementable generically); end-of-answer
token ids (the `END` set, §5.3); and a sanity prompt with an expected top label and
minimum top-label credence (used by the opt-in e2e suite, §14). The
registry must include at least one `</think>`-family model (e.g. a small Qwen3) and
one Harmony-family model (gpt-oss) so both glue paths stay exercised. Unlisted
models work via detection + warning; users can register their own entries.

## 10. Backends

One capability contract: a backend supplies raw logits for any requested token
ids at any supplied token context, using an accessible tokenizer consistent with
the model. V1 implements this with MLX on Apple Silicon.

Physical locality is not the contract. A future backend could run on an NVIDIA
cluster or a researcher-controlled server if it preserves the same token-level
access. Restricted top-K probability APIs are outside the core; there is no
second protocol, partial-readout result, or hosted-provider release milestone.
Consider them only in response to a concrete future need
(`POSSIBLE_FUTURE_EXTENSIONS.md` §9).

### 10.1 `Model` and `load_model`

```python
model = load_model("mlx-community/Qwen3-4B-4bit")  # backend="auto" selects MLX in v1
model = load_model("/shared/models/Qwen3-4B-4bit") # complete local snapshot; no download
```

The first argument accepts a model repository identifier or a filesystem path
to a complete snapshot. A path performs no network fetch, so workers can load
from shared or node-local storage (§13). Unsupported backend choices raise a
clear error; never fall back to a restricted probability API.

`Model` bundles the backend, its tokenizer, and the resolved `ReasoningProfile`
(bypass insert, native-disable kwargs, and profile source). Model loading resolves
these resources but does not drive measurement or generate text.

### 10.2 Full-logits protocol

```python
class FullLogitsBackend(Protocol):
    def tokenizer(self) -> Tokenizer: ...

    def next_token_logits(
        self, token_ids: Sequence[int], wanted: Sequence[int]
    ) -> TokenScores:
        """Raw logits of every requested token at the next position.

        Values are float32 model logits. `wanted` restricts the returned values,
        not which logits the model is capable of exposing. Every requested id
        must be present; the backend never samples or truncates to top-K.
        """

@dataclass(frozen=True)
class TokenScores:
    logits: Mapping[int, float]   # every requested id
    log_normalizer: float         # logsumexp of the logits over the whole vocabulary
```

`log_normalizer` costs one reduction over logits the model has already computed.
credences uses it only for coverage at the scoring position (§6.5); it never
enters credences. It must be finite and at least the logsumexp of the returned
requested logits, allowing float32 roundoff; otherwise raise
`BackendReadoutError`.

A missing requested id is a backend contract violation: raise
`BackendReadoutError`, not a zero probability. Reject NaN or positive infinity
with the same error. Negative infinity represents zero weight when any allowed
score is finite; an all-`-inf` allowed set triggers uniform token fallback (§5.1).
The backend returns the raw scores; masking, forcing, and fallback belong to
the pure measurement logic, not a backend-specific repair.

The tokenizer adapter exposes:

```python
class Tokenizer(Protocol):
    def encode(self, text: str, /, *, add_special_tokens: bool) -> list[int]: ...
    def decode(self, ids: Sequence[int], /, *, skip_special_tokens: bool,
               clean_up_tokenization_spaces: bool) -> str: ...
    def apply_chat_template(self, messages, /, *, tokenize: bool,
                            add_generation_prompt: bool,
                            continue_final_message: bool,
                            **chat_template_kwargs): ...
    eos_token_id: int | None
    chat_template: str | None
    additional_special_tokens: list[str]
```

The protocol declares no defaults. Tokenizer libraries disagree on them (Hugging
Face adds special tokens and tokenizes chat templates by default), so every call
passes each flag explicitly and any tokenizer accepting these keywords conforms.

The model profile chooses compatible rendering flags; do not open a new assistant
turn and continue a final message simultaneously. The adapter preserves the
special-token policy required by §8. Its tokenizer-bound profile must keep an
optional final assistant prefill exactly (§8.1) and implement the validated
in-context encoder (§8.2). Neither operation belongs in the backend protocol.

The Credoscope normalizes only over allowed tokens. Beyond the one
`log_normalizer` value, no full-vocabulary softmax or transfer of the full
vocabulary distribution to the caller is required.
No `generate`, `create_cache`, `copy_cache`, or low-level `forward` methods are
part of this protocol. Intra-measurement KV reuse is backend-internal; its request
lifetime is an implementation review item (§13, §16).

**MLX reference:** use `mlx-lm`, returning requested final-position logits cast
to float32. Begin with a correct uncached implementation, then verify any
intra-measurement prefix reuse against it. No reasoning generation in v1.

### 10.3 Additional full-access backends

An NVIDIA backend is deferred until the local instrument is correct and useful.
Choose its runtime when implementation begins, using the same full-logits
contract and parity tests. A future server adapter must support supplied token
contexts and arbitrary requested scores; merely returning logprobs for generated
text is not sufficient. Scheduling and weight staging remain external (§13).

## 11. Task prompts and semantic mappings are external

The Credoscope scores exact candidate strings against prompts supplied by the
caller. It does not construct task prompts, randomize option order, assign codes, or
map candidate strings back to semantic labels. Those operations have different
research assumptions and should not complicate the measurement primitive.

The caller also owns any final assistant prefill, supplied through `messages=`.
The library adds no answer cue. It owns only candidate trimming and the
two-spelling policy (§4.1, §4.3, §8), keeping native model formatting separate
from research instructions, semantic descriptions, and randomized code meanings.

For a coded prompt, a caller might build a Credoscope over `["A", "B", "C"]`,
retain a per-input mapping such as `{"A": "neutral", "B": "positive", "C":
"negative"}`, and remap the returned candidate credences after measurement. Every
persisted result should carry a stable input identifier and that mapping; large or
distributed runs should not rely on positional joins alone.

Researchers concerned about presentation effects can compare candidate orderings
or balanced code assignments in their prompt-generation layer. A possible
standalone helper and its metadata contract are recorded in
`POSSIBLE_FUTURE_EXTENSIONS.md`; it may eventually live in this package or in a
separate package, but it is not part of v1.

## 12. Architecture

Use a **uv-managed library project**, initialized with uv's `--lib` template,
with a `src` layout and typed-package marker. Configure the planned Hatchling
build backend in `pyproject.toml`; uv manages dependencies, the environment,
lockfile, development commands, and package builds. The repository root contains
`pyproject.toml`, committed `uv.lock`, `README.md`, and `tests/`.
The library source tree is:

```
src/credences/
  py.typed           # typed library marker
  __init__.py        # build_credoscope, Credoscope, Measurement, RawReadout,
                     # load_model, Model, choose_top_label, top_label_weights
  credoscope.py      # Credoscope + build_credoscope; shared preparation for preview/measure;
                     # runs trie plans against a backend;
                     # the ONLY module that drives backends
  probabilities.py   # pure: result dataclasses + §5 math, summaries, and tie helpers
  trie.py            # pure: §7 trie over every spelling path, readout plans, validation
  types.py           # pure: Message + ChatSpec types
  tokenization.py    # tokenizer-bound, backend-free: §8 boundary contract + Tokenizer protocol
  reasoning.py       # tokenizer-bound, backend-free: ReasoningProfile, detection, blessed registry
  model.py           # Model + load_model (backend selection; profile resolution)
  errors.py          # CredencesError, TokenBoundaryError, BackendCapabilityError,
                     #       BackendReadoutError
  backends/
    base.py          # FullLogitsBackend protocol
    mlx.py           # v1 reference backend
  frames.py          # OPTIONAL: list[Measurement] -> pandas/polars (optional extras)
```

Three dependency tiers, and the import-linter check in CI enforces the first rule:

1. **Pure** (no tokenizer, no backend): `trie`, `probabilities`, `types`, `errors`.
2. **Tokenizer-bound, backend-free** (HF-tokenizer objects, no runtime imports):
   `tokenization`, `reasoning`, `model` (protocol-typed).
3. **Backend-bound**: `backends/*`, and `credoscope.py` as the single chokepoint
   that drives a backend.

## 13. Execution, caching, and later optimization

V1 ships **no cross-call inference/KV cache**. Loaded model weights and tokenizer
objects are reused across calls; prompt-dependent state and scores are not.
Reusing an immutable trie plan (§8.3) is a separate, pure optimization.
Within one measurement, the backend may reuse context and trie-prefix KV state,
but only with an explicit request lifetime and cleanup on both success and error.
Until that internal lifecycle is reviewed, recomputing each requested context is
the correct baseline. No low-level cache operations belong in the public API.
Cross-call inference-state caching is deferred pending profiling.

Every measurement call is independent and deterministic when the backend is
deterministic. There is no global progress state, scheduler state, or mutable
cross-call accumulator. Distributed corpus runs therefore shard outside the
package: a worker receives a small independently retryable shard, loads the model
once, builds one Credoscope, measures the shard, and persists its results. A
preempted shard can be rerun without coordinating with other workers.

Model weights should be pre-staged in a shared cache reachable over the local
network or copied once to node-local storage during job setup. Workers load from
that path rather than downloading weights from the internet. SLURM or the
consuming application owns shard manifests, placement, retries, and completed-shard
tracking. This keeps shards small enough to schedule opportunistically without
making model setup the dominant cost. Future backend batching may accelerate the
sequence form of `.measure()` without changing these semantics.

**Optimize later via swappable, parity-tested components:** the pure modules have
stable interfaces and property tests pinning their outputs; a hot function can be
reimplemented and must pass the same property tests **plus a
golden-parity test** (identical inputs → outputs within FP tolerance of the Python
reference) before being trusted, then benchmarked. Measure tokenization, model
prefill, and trie traversal separately before selecting an optimization.

One candidate optimization, only if profiling shows branch readouts matter:
skip the readouts below a first token whose probability at the scoring position
is under a small tolerance (for example 0.1%), splitting its mass as the
candidate's other spelling splits. The error is bounded by that tolerance, and
the measurand in §5.1 stays fixed; a tolerance must never decide which spellings
count.

## 14. Testing

Three tiers; every test guards a named failure mode — no tests written just to
raise the test-coverage number.

**Tier 1 — pure property tests (Hypothesis + a thin fake backend).** The fake
backend implements `FullLogitsBackend` over a ~10-token vocabulary with declared
logits. Properties:

- **Summed-spelling parity (the load-bearing one):** trie scoring equals an
  independent reference that enumerates every path, applies the naive chain
  rule, and sums each candidate's paths. Include the §5.2 and §8.4 shapes, paths
  shared by both spellings, and spellings that start with the same token.
  Keeping only the stronger spelling, or normalizing per candidate before
  merging shared first tokens, must fail parity.
- Shared edges count once; a path shared by a candidate's two spellings counts
  once; cross-label path collisions, empty paths, and a candidate's path that
  prefixes its own other path raise. Candidate iteration order cannot alter
  credences beyond floating-point tolerance.
- Credences sum to 1 within FP tolerance (per-node masked probabilities over the
  allowed set sum to 1 by construction; log-space accumulation adds rounding);
  an all-`-inf` allowed set yields `1 / len(allowed)` per distinct token. Mixed
  finite/`-inf` scores give zero to `-inf` tokens; finite scores normalize normally.
  Singleton transitions are forced even when a fake backend would score them
  `-inf`, and need no query. Test token-uniform but label-nonuniform outcomes.
- End-vs-continue: `p̃(END) + Σ p̃(continuations) == 1` at the node; the §5.3/§7.1
  `END`-disjointness `ValueError` fires on an overlapping pair; a prefix overlap
  with no resolvable `END` fails path validation before inference.
  All-`-inf` fallback with multiple end ids aggregates their token-uniform mass.
- Log-space robustness at extreme logits, including explicit uniform fallback
  without `-inf - (-inf)`: finite log credences remain finite
  even when exponentiation underflows to zero. JSON serialization distinguishes
  finite logs from exact `-inf` (serialized as `null`).
- Exact ties return all winners in candidate order, with no randomization or
  rounding-induced ties. Caller-seeded tie selection only returns winners;
  fractional weights sum to one and are distinct from the credence vector.
- Candidates are trimmed at both ends only; internal whitespace is kept; result
  keys stay as supplied. Candidates that are empty or duplicated after trimming,
  such as `"positive"` and `" positive"`, raise before scoring.
- Coverage equals the full-vocabulary probability of the allowed first tokens,
  is always defined (a singleton scoring position is still read), and does not
  change credences. An invalid `log_normalizer` raises `BackendReadoutError`.
- Construction rejects fewer than two candidates; scalar input returns `Measurement`,
  sequence input preserves order and length, and empty sequence input returns `[]`.
- Preview returns `str` or aligned `list[str]` under the same input dispatch;
  preparation errors match measurement, no backend calls occur, and reports show
  the same paths and readout plan that measurement uses. No automatic
  print/log/file side effects.
- Missing requested backend scores or invalid numeric scores raise
  `BackendReadoutError`, never silently renormalize an incomplete readout.

**Tier 2 — tokenizer-only tests (real tokenizers, no model weights).** Gated on
network/fixture availability, not on hardware — this is the highest-value
regression surface and must run in ordinary CI. Per blessed-registry entry: §8.2
in-context encoding with exact round trips for both spellings; the scoring
position, including a trailing-space prefill that merges into the answer and
candidates that merge differently (`"yes"` with `"1"`); the §8.4
prompt/message/prefill cases, including templates that trim the prefill; no
doubled beginning-of-sequence token; bypass inserts and native-disable kwargs;
different whole-spelling segmentations and `END` validation. Reject unsupported
rendering/encoding before calling a model. Preview/measurement preparation
parity and faithful report rendering (§4.4) must cover these same cases,
including a user message that itself ends in `"Answer:"`. With fake scores on
those real paths, test cold/warm/rebuilt plans.

**Tier 3 — opt-in real-model e2e (`CREDENCES_RUN_MLX_TESTS=1`).** Against the
blessed registry's sanity prompts: expected top label and minimum top-label credence,
bypass sanity on at least one `</think>`-family and one Harmony-family model.

Golden fixtures: a hand-built fake-backend scenario per interesting shape —
two single-token candidates with unequal scores and exact ties;
the `"positive"`/`"position"` example (§8.4);
shared-prefix multi-token (the §5.2 five-candidate
example, with its hand-computed credences); prefix overlap with an
end-vs-continue node (`"increase"`/`"increase substantially"`, expected credences
including the `p̃(END)` factor); candidates `"A"`/`"AB"` **declared as `[t_A]` and
`[t_A, t_B]` in the fake vocabulary** (an end-vs-continue fixture, plus the
`END`-disjointness `ValueError` fixture when a continuation token is placed in
`END`) — with exact expected credences computed by hand and stored
in-repo. These are also the §13 golden-parity reference for any future
reimplementation.

## 15. Distribution

- **PyPI:** planned name `credences`; recheck availability before publishing.
  Public, MIT. This planning document does not reserve a package name.
- **Project tooling:** uv-managed library (`uv init --lib`), Python >= 3.14,
  `src/credences/` and `py.typed` (§12). Keep Hatchling as the configured build
  backend. Use `uv add`, `uv sync`, `uv run`, and `uv build`; commit `uv.lock`
  for development/CI and use `uv sync --locked` there. The lockfile does not pin
  downstream users' environments. See [uv library projects](https://docs.astral.sh/uv/concepts/projects/init/#libraries).
- Keep core dependencies minimal. Developer tools belong in the `dev` dependency
  group; `mlx` / `pandas` / `polars` remain optional runtime extras.
- README: the §1 pitch; the forced-choice measurand (§5.1); the §5.5 token-boundary
  and §9 estimand caveats; the §6.1 confidence summaries; and citations to CDS
  (NBER w34965) for the binary measure and forced-choice logprob method and
  Titelbaum for the term.

## 16. Implementation checkpoints and deferred scope

- **Internal backend request lifetime:** specify how one measurement owns,
  branches, and releases KV state before implementing prefix reuse. Include
  failure cleanup; keep the uncached scoring baseline available (§10.2, §13).
- **Model loading and dependency tiers:** choose the backend-construction entry
  point before wiring `load_model`. A protocol-typed `model.py` cannot also import
  runtime implementations while claiming to be backend-free (§12).
- **Supported-model validation:** verify in-context encoding of both spellings
  with real tokenizer fixtures before declaring support. Validate native
  assistant opening, exact prefill rendering (including restored trailing
  whitespace), and exact round trips. A failed fixture calls for an explicit
  profile decision, never rewriting prompt text or silently dropping a spelling.
- **Broader variants / synonym sets:** v1 scores only the bare and
  single-ASCII-space spellings in their canonical tokenizations. A possible
  extension lets callers list accepted spellings per candidate (for example
  `"Yes"` and `"yes"`), summed the same way; newline/tab variants and
  non-canonical tokenizations stay out. See `POSSIBLE_FUTURE_EXTENSIONS.md` §10.
- Generated reasoning, post-reasoning measurement, post-trace prefill interactions, and
  multi-sample averaging are deferred to `POSSIBLE_FUTURE_EXTENSIONS.md` §8.
  V1 supports only `reasoning="bypass"`.
- Additional backends must satisfy §10; platform-specific guided generation alone
  does not establish arbitrary-context logit access.
- Smallest useful summary set beyond top-label / entropy / margin confidence
  (top-2 ratio? Gini?). Resist until a consumer asks.

## References

- M. Titelbaum, *Fundamentals of Bayesian Epistemology I: Introducing Credences*, OUP 2022.
- Stanford Encyclopedia of Philosophy, "Bayesian Epistemology".
- H. Chen, A. Didisheim, L. Somoza, *Out of the Black Box: Uncertainty Quantification
  for LLMs via Conditional Probabilities*, NBER Working Paper 34965, 2026 ("CDS").
- Park et al., [*Grammar-Aligned Decoding*](https://arxiv.org/abs/2405.21047),
  NeurIPS 2024.
- R. L. Geh, H. Zhang, K. Ahmed, B. Wang, G. Van den Broeck,
  [*Where is the signal in tokenization space?*](https://arxiv.org/abs/2408.08541),
  EMNLP 2024.
- S. Lundberg, M. T. Ribeiro, [*The Art of Prompt Design: Prompt Boundaries and
  Token Healing*](https://github.com/guidance-ai/guidance/blob/main/notebooks/art_of_prompt_design/prompt_boundaries_and_token_healing.ipynb), 2023.
- EleutherAI, [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
  `_encode_pair` in `lm_eval/api/model.py`.
- "On the attribution of confidence to large language models," *Inquiry*, 2025 —
  terminology-in-ML for credence attribution.
