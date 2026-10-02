# credences — Specification (v1)

> **Status:** seed specification for a fresh `credences` repository.
>
> **Reading order:** read this file §1–§8 first. Then `DECISIONS.md`
> for *why* the design is shaped this way, then `AGENTS.md` for the working rules.
> **Review focus:** tokenize the candidate and one ASCII-space-prefixed form
> (§4.3, §8), select one by its first-token score, then score the selected trie. The caller
> owns optional assistant prefills through `messages=`; context tokens stay fixed.
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
opens the answer position; the library adds no answer cue. For each candidate,
the two possible continuation texts are the candidate itself and one ASCII space
followed by that candidate. For each candidate, select the complete path whose
first token has the higher logit at the fixed context; exact ties select the bare
form. Build and score a trie containing only the selected paths (§5, §8).
Result keys remain the declared candidates.

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
| **candidate** | an exact answer value (`"positive"` or `"A"`); first-token scores select its bare or single-space-prefixed path, and it remains the result key | — |
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
- Two canonical continuation forms per candidate: bare and prefixed with one
  ASCII space. First-token scores select one complete path per candidate before
  constrained scoring (§4.3, §5, §8).
- Native assistant opening by default; optional caller-owned assistant prefill
  through `messages=` (§4.2). No package-supplied answer cue or template option.
- A no-inference `.preview()` of the actual compiled context and candidate
  possible continuations, for checking formatting before measurement (§4.4).
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
  mappings remain caller-owned; the library defines the two-form policy (§4.3).
- ❌ Restricted top-K probability APIs, partial readouts, or imputed token scores.
- ❌ Whitespace variants beyond the one optional initial ASCII space, case or
  synonym expansion, enumeration of alternative tokenizations, or mandatory JSON.
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
each measurement prepares both context-dependent paths, queries first-token
scores to select one per candidate, and then scores the selected trie (§8).

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

### 4.1 `candidates` — exact answer strings (bound at build)

`candidates` is a `Sequence[str]`, for example
`["positive", "negative", "neutral"]` or `["A", "B", "C"]`. Each string is
an exact answer value and the corresponding key in `Measurement.credences`.
Each candidate has two continuation texts: `candidate` and `" " + candidate`
(§4.3). The Credoscope has no separate knowledge of semantic labels,
descriptions, codes, or option presentation in the task prompt.

Candidates must be unique, non-empty strings. Sequence order is the canonical
candidate order used to present results, including all tied winners (§6.4),
without selecting one winner. A bare string and all mapping types are rejected;
candidate descriptions and semantic metadata are not accepted here.
Candidate text is literal, including any internal whitespace. If the accepted
forms of two candidates overlap, reject the set as ambiguous; for example,
`["positive", " positive"]` would assign the same response to two result keys.
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

The prefill is supplied text, not generated reasoning. Preserve its content,
including trailing whitespace. Place it at the final answer position after the
model-specific reasoning bypass. If a profile cannot render that combination
faithfully, raise before scoring; do not relocate, trim, or silently drop it (§8).

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
fixed context text, with no chat template, added cue, or implicit newline. The
caller includes any desired prefill in that string. The same two continuation
forms and fixed-token-boundary rules apply. On a chat model, use `messages=` for
an assistant prefill rather than embedding chat markers in `prompt=`.

### 4.3 Two canonical starting forms and selection

For each candidate `c`, v1 prepares exactly these two possible surface forms:

```text
bare:   c
spaced: " " + c
```

The added character is one ASCII space (`U+0020`), not arbitrary whitespace. It
occurs only before the entire candidate. Do not add tabs, newlines, capitalization
variants, synonyms, or independently optional spaces at internal token boundaries.
There is no formatting argument, placeholder syntax, or automatic answer cue.
Prefills belong to the input (§4.2).

Tokenize each complete surface once using the profile's canonical continuation
encoder at the same fixed context (§8). Illustrative splits include:

| Candidate | Bare path | Spaced path |
|---|---|---|
| `"positive"` | `["positive"]` | `[" positive"]` |
| `"very positive"` | `["very", " positive"]` | `[" very", " positive"]` |
| `"elequent"` | `["ele", "quent"]` | `[" ele", "quent"]` |
| `"eloquent"` | `["elo", "quent"]` | `[" eloquent"]` |

These splits are illustrative, not promises about a vocabulary. The leading
space can change more than the first token boundary, so encode both whole strings
rather than replacing the first token id. After choosing a form, its internal
tokenization is fixed. Do not also enumerate paths such as `["p", "os", "itive"]`
unless that is itself the canonical encoding of one of the two surfaces.

At measurement time, query the logits of all distinct first-token ids at `C` in
one readout. Independently for each candidate, select the form whose first token
has the higher score. Comparing logits gives the same selection as comparing
full-vocabulary probabilities at that context. For an exact tie, including two
`-inf` scores, choose the bare form. This deterministic spelling tie-break is
separate from reporting all tied final labels (§6.4).

Retain the selected form's **entire** token sequence. Build the scoring trie from
these selected paths, merging shared prefixes. A shared first token enters the
root normalizer once, not once per candidate. Do not sum or average the forms,
multiply in a probability of choosing a form, or choose the maximum complete-path
probability. The selection criterion is only the first-token score (§5.1).

Vocabulary searches for whitespace variants of the bare first piece are not
sufficient: if `"eloquent"` is `["elo", "quent"]` but `" eloquent"` is one token,
matching only `"elo"` and `" elo"` misses the latter. Tokenize whole forms instead.
Other whitespace tokens can exist, including standalone indentation/newline tokens
and tokens combining a tab with text. Their existence does not expand the two-form
policy: no search over `\s*`, tab/newline prefixes, or multiple added spaces in v1.

The first-token model readout receives only the unchanged context `C`, including
any prefill ending in `"Answer:"` or `"Answer: "`. It does not receive either
candidate form appended to the prompt. The prepared paths identify which
next-token logits to inspect in that one readout. If `"positive"` and
`" positive"` are each one token, compare those two logits and retain the
higher-scoring token for the label `"positive"`. If either form spans multiple
tokens, compare their first-token logits and retain the selected complete path.
Then compute credences by constrained scoring of the selected trie, as above.

Prefill whitespace remains part of `C`; it is not removed or deduplicated against
a candidate's leading space. Thus the spaced form adds one space even when the
prefill already ends in a space. Native template or prefill newlines likewise
remain in the context, not among generated answer alternatives.

The policy is a bounded, model-selected measurement constraint, not a claim to
capture every way the model could emit a label, maximize whole-answer likelihood,
or eliminate prompt sensitivity. Use `.preview()` to inspect the fixed context
and both possible paths; the selected path is known only after a score readout.

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

- The full decoded fixed context `C`, without truncation or hidden special tokens.
  This includes actual role delimiters, bypass text if present, and any supplied
  assistant prefill. Show both readable text and an escaped representation so
  trailing spaces and newlines can be inspected.
- Every candidate in declared order, with both surfaces labeled `bare` and
  `spaced`, their escaped text, and their complete token-id paths. Derive each
  continuation text from `decode(C + T)` after the verified `decode(C)` boundary,
  not from isolated token decoding (§8.2). Context plus continuation reconstructs
  each complete candidate-filled text. Explain that first-token scores select one
  complete path per candidate during measurement; do not mark either as selected.
- The resolved `END` token ids when a possible end-vs-continue decision needs
  them (§5.3), without claiming that the selected trie will contain that branch.

Label context and continuations separately. Candidates are supplied alternatives,
not generated answers or predictions. The report is for inspection, not a second
prompt to feed back into `.measure()` or a machine-readable serialization format.
Later branch readouts use `C` plus a candidate-path prefix (§7); the complete
candidate-filled texts are not all independently submitted for generation.

**One preparation path:** preview and measurement must share input validation,
native rendering, fixed-context continuation encoding, and potential-path/`END`
validation. There is no simplified preview renderer. For identical input and
scope, preview describes exactly the `C` and alternatives measurement will
consider. It cannot know the selected paths, actual branch plan, logits, or
credences without inference. Invalid inputs or boundaries raise the same
preparation errors; no partial batch report is returned. Backend/score errors
can occur only during measurement.
Preview does not query logits, generate text, allocate model/KV state for a call,
or change subsequent measurements. It uses the already-built scope's tokenizer
and configuration; this does not imply a separate model-loading API.

Return the report without printing, logging, or saving it automatically. Do not
remove duplicate cues, infer prompt intent, or warn based on wording heuristics.
The purpose is to let the researcher inspect their own full inputs, not to add
prompt metadata to every `Measurement`.

## 5. What is measured - the math

This section is normative. Every number the library returns is defined here.

### 5.1 The measurand: selected-form constrained distribution

The procedure has two stages: **deterministically select one representation per
candidate using first-token scores**, then measure the forced-choice distribution
over the selected token paths. It is not the distribution of a decoder admitting
both forms at once. It does not sum their mass, average their distributions, or
maximize complete-answer probabilities.

Let `B(c)` and `W(c)` be the canonical paths for candidate `c` and `" " + c`,
both extending the same fixed context `C` (§8). Query the raw model score `z(t)`
at `C` for every distinct first token in those paths. Select:

    T(c) = W(c)  if z(W(c)[0]) > z(B(c)[0])
           B(c)  otherwise

Thus ties, including two negative infinities, select the bare form. Reuse these
scores for the root of the selected trie; selection is not another random event
and contributes no probability factor. The resulting trie can depend on the
prompt's scores even when all possible token paths are unchanged.

Build one trie containing exactly `T(c)` for each candidate. At prefix `P`,
the allowed set `A(P)` contains the distinct next tokens of selected paths,
plus the model's `END` tokens when a selected path ends there and another
continues (§5.3). At a leaf, stop without an additional model query.

For a nonempty allowed set of size `n = |A(P)|`, define the transition rule:

    p_tilde(t | P) = 1                              if n = 1
                    1/n                            if all allowed scores are -inf
                    exp(z_P(t) - logsumexp(z_P(A))) otherwise

A singleton is forced, regardless of its unqueried model score. At a branching
node with at least one finite score, individual `-inf` scores get probability
zero and the finite scores divide the total mass. NaN, positive infinity, and
missing requested scores are errors, not inputs to the uniform fallback (§10.2).

The all-`-inf` case is an explicit uniform **token-level fallback**, not an
inference that the model has equal beliefs over labels. A nested trie need not
yield uniform final label credences when its branch transitions are uniform.
Never replace small finite scores with this fallback.

A candidate's credence is the product along its one selected path:

    credence(c) = product of p_tilde(next token | prefix) along T(c)
                  * p_tilde(END | T(c)) if a longer selected path continues there

Here `p_tilde(END | P)` is the sum of transition probabilities for the distinct
tokens in `END`. A shared edge contributes once to its node's normalizer, even
when many candidates use it. Credences sum to 1 by construction, with no
label-level renormalization.

**Interpretation:** a decoder with this same first-token selection, selected-path
mask, forced transitions, and uniform fallback would sample this distribution.
It is not a promise of parity with arbitrary structured-output engines or of
prompt-independent, calibrated belief. Discarded forms can have real probability;
max selection is a specified representation policy, not a noise-removal theorem.

**Why this is not global conditioning:** computing each complete answer's joint
unconstrained probability and renormalizing at the end is a different target.
Local masks redistribute mass at each decision, forced steps contribute 1, and
the selected output language itself depends on the initial scores. See
*Grammar-Aligned Decoding* (References) for the distinction between local masking
and conditioning. Full joint probabilities would also require full-vocabulary
normalizers at every step, which requested raw logits alone do not provide.

### 5.2 Worked example (five candidates, selected forms)

Candidates `["very negative", "negative", "neutral", "positive", "very positive"]`.
Assume these illustrative root probabilities:

| Starting piece | Bare-token probability | Spaced-token probability | Selected token |
|---|---|---|---|
| very | 0.10 | 0.001 | `"very"` |
| negative | 0.65 | 0.002 | `"negative"` |
| neutral | 0.20 | 0.001 | `"neutral"` |
| positive | 0.02 | 0.001 | `"positive"` |

Both `"very ..."` candidates select the path starting with `"very"`. The root
normalizer is `0.10 + 0.65 + 0.20 + 0.02 = 0.97`, with `"very"` counted once.
At the selected `"very"` prefix, let `p(" negative") = 0.95` and
`p(" positive") = 0.01`, giving local denominator 0.96.

```text
credence(negative)      = 0.65/0.97                 = 0.6701
credence(neutral)       = 0.20/0.97                 = 0.2062
credence(positive)      = 0.02/0.97                 = 0.0206
credence(very negative) = (0.10/0.97) * (0.95/0.96) = 0.1020
credence(very positive) = (0.10/0.97) * (0.01/0.96) = 0.0011
                                               sum = 1.0000
```

There are two readouts: the initial comparison of all eight distinct first-token
ids, reused to score the selected root, and the selected `"very"` branch.
There is no readout after the discarded `" very"` token.

*End-vs-continue example.* If selected paths for `"increase"` and
`"increase substantially"` are `["increase"]` and
`["increase", " substantially"]`, query `END union {" substantially"}` there.
If their selected forms instead begin with distinct tokens and neither selected
path prefixes the other, that end-vs-continue branch is absent.

### 5.3 The end-of-answer set (`END`)

An end-vs-continue decision is needed when a selected path is a strict prefix of
another candidate's selected path. Alternatives of the same candidate never
coexist in the scoring trie and do not create such a decision.

Resolution order: explicit `end_tokens=` override on `build_credoscope` ->
blessed-model registry (§9.2) -> for unlisted models, `{tokenizer.eos_token_id}`
plus additional special tokens matching a known end-of-turn list, with a warning.
Resolve and deduplicate `END` at build time.

Preparation conservatively validates every potential prefix overlap between
forms of different candidates, before model inference. If any such pair lacks
a usable `END`, or its continuation token is also in `END`, raise `ValueError`.
Preview performs the same preflight. After selection, only overlaps that remain
in the selected trie produce readouts. This ensures inference cannot choose an
unsupported stopping boundary.

Normalize over individual token ids, including all distinct `END` ids, then sum
the `END` probabilities. If all scores are `-inf`, each allowed token gets
`1 / |A(P)|`: with two end tokens and one continuation token, stopping gets
`2/3`, not `1/2`. Uniform fallback is over tokens, not over grouped outcomes.

**Do not implement:** crediting the shorter candidate with
`1 - sum p(candidate continuations)` over the full vocabulary. That would count
unrelated words and punctuation as choosing to stop. Only actual `END` tokens
represent stopping at an end-vs-continue branch.

### 5.4 Failure modes these definitions exclude

1. **Scoring the unselected union.** Comparing both first tokens does not admit
   both complete forms into the scoring trie. Discard the losing path first.
2. **Per-label root normalization.** If two selected paths begin with `"pos"`,
   that token gets one root weight; divide it at a later branch. Counting it
   twice or normalizing per-candidate first-token maxima changes the result.
3. **Replacing only the first token.** A selected `" positive"` token must not
   inherit `"itive"` from a discarded `["pos", "itive"]` path.
4. **Mixed probability conventions.** All actual decisions use §5.1's rule,
   including end-vs-continue nodes. No unnormalized terminal factors or
   complement terminal mass.
5. **Zero-score arithmetic.** Do not evaluate `-inf - (-inf)`, replace tiny
   finite scores by zero, or use uniform fallback when any allowed score is finite.

The §14 parity test compares the complete two-stage procedure with an independent
reference: select each path, derive the selected allowed sets, then apply the
chain rule including forced transitions and all-`-inf` fallback.

### 5.5 Numerical rules

- Compare raw first-token logits at the same fixed context, not rounded
  probabilities. Equal computed scores select the bare form, independently of
  candidate order. Reuse the same readout for selected-root normalization.
- Compute in **natural-log space**. With at least one finite allowed score,
  step logprob is `score(t) - logsumexp(allowed scores)`. A forced step contributes
  0; an all-`-inf` branching step contributes `-ln(n)` for each of its `n`
  tokens. Implement these cases explicitly. An `END` event uses `logsumexp`
  over its normalized token logprobs, including in the fallback case.
- A candidate's log credence is the sum of step logprobs along its selected path.
  There is no sum across forms. For `"very negative"` in §5.2:

      log credence = ln(0.10) - ln(0.97) + ln(0.95) - ln(0.96)

  Exponentiating gives about `0.1020`. Preserve a finite log credence even when
  exponentiation underflows to `0.0`; exact zero mass has log credence `-inf`.
  Determine winning labels from unrounded log credences (§6.4).
- Allowed-set log-softmax uses **float32**, casting up bf16/fp16 logits first.
  Avoid forming full-vocabulary probabilities before extracting allowed scores.
  Quantized weights still measure the loaded quantized model.
- A fixed absolute probability error can be large relative to a small probability.
  Log-space arithmetic avoids additional underflow; it cannot undo upstream
  quantization/logit error. Max selection does not guarantee better accuracy or
  establish that the discarded form was noise.
- Entropy uses `0 * log 0 = 0`.
- Measurement has no sampling or cross-call semantic state. Bitwise reproducibility
  is not promised across kernels, batching, devices, or environments. Researchers
  own experiment tracking.
- **Token-granularity caveat:** first-token selection can compare `"pos"` with
  `" positive"`, even though they consume different amounts of text. A prefix
  token's score is not the probability of the entire label. A selected standalone
  space can likewise carry broad continuation mass. This is part of the specified
  token-level procedure, not semantic equivalence or calibration. Inspect both
  possible paths with preview and the selected paths in `raw.scored_token_ids`.
- Selection is discrete: a small first-logit change can switch a whole path and
  its later branch contexts. Determinism does not imply insensitivity to numerical
  perturbations near a form-selection tie.

## 6. Result objects

```python
@dataclass(frozen=True)
class RawReadout:
    credence_logprobs: dict[str, float]    # per candidate: ln(credence) — natural log; the
                                           # log-space source of Measurement.credences
    scored_token_ids: dict[str, tuple[int, ...]]
                                           # per candidate: selected complete path;
                                           # does not list discarded form comparisons
    context_token_count: int                # length of C actually scored

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
- Experiment metadata, model identifiers, versions, and semantic mappings live
  in caller-owned data, not a per-measurement metadata object. There is no
  `Measurement.reasoning`: bypass is the only supported mode. The raw readout
  contains measurement-specific log credences and token paths, not run metadata.

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

## 7. The token trie

The trie is pure and backend-independent: selected token paths in, a readout plan
out, credences back. A separate pure selector compares supplied first-token scores;
the Credoscope owns the model calls and invokes selection before trie construction.

### 7.1 Inputs and structure

Input: `sequences: dict[str, tuple[int, ...]]`, exactly one selected complete
path per candidate. All paths extend the same fixed context `C`.

Each node is a unique prefix `P`, with distinct child token ids and at most one
candidate owning the terminal. Merge shared prefixes; never duplicate an edge's
weight because several candidates follow it. A terminal with children is an
end-vs-continue node, whose allowed set includes `END`.

Validation:

- Empty paths and identical paths owned by different candidates raise `ValueError`.
- At an end-vs-continue node, `END` must be nonempty and disjoint from child ids.
- Preparation additionally validates both possible forms before selection (§8.2),
  so a backend readout cannot reveal a previously unvalidated path.
- A candidate contributes only its selected path. Same-candidate form deduplication
  and path aggregation are not operations of this scoring trie.

### 7.2 The readout plan

The Credoscope first queries all distinct first tokens of both forms at `C`,
then selects the paths and builds the trie. It reuses that readout to normalize
the root over **selected distinct child ids only**.

For each non-root node `P` with at least two allowed token ids, query
`next_token_logits(C + P, A(P))`. Skip singleton nodes: their transition is
defined as forced, with probability 1, even if the unqueried score would be
`-inf`. Leaves stop without querying an end score.

Total readouts are one initial selection readout plus the number of non-root
branch nodes in the selected trie. Even if the selected root has only one child,
the initial readout can be necessary to choose that child's spelling and hence
later model contexts. Forced prefixes must still be appended before scoring a
later branch; skipping a readout does not delete tokens from model history.

With `["very positive", "very negative"]` and shared first-token alternatives
`"very"` / `" very"`, selection chooses the same form for both candidates.
Read the two first-token scores once, then score only the selected prefix's
`" positive"` / `" negative"` branch. If all selected first tokens are distinct,
there is only the initial readout. Forward-pass cost remains a backend concern;
readout count alone is not a wall-clock guarantee.

### 7.3 Scoring

Given selected paths and validated requested-token logits:

```text
for each node P whose allowed set A(P) is nonempty:
    if len(A(P)) == 1:
        step_logprob(P, sole token) = 0       # no score query
    elif all scores over A(P) are -inf:
        step_logprob(P, each token) = -ln(len(A(P)))
    else:
        logdenom = logsumexp(scores over A(P))
        step_logprob(P, token) = score[P][token] - logdenom

for each candidate c with selected path T(c):
    log_credence(c) = sum of step logprobs along T(c)
    if another selected path continues through T(c):
        log_credence(c) += logsumexp(
            step_logprob(T(c), e) for e in END
        )
```

The normalized transitions and stopping events partition mass among candidates,
so final credences sum to 1 without another normalization. A selected path with
a zero-probability edge retains log credence `-inf`; uniform transitions farther
down that path do not restore its lost incoming mass.

**Parity invariant (§14):** selection plus trie scoring must match an independent
reference using the same deterministic selection rule and naïve chain rule over
the selected paths. Include shared roots, score-dependent branch appearance,
singleton forcing, mixed finite/`-inf` scores, and uniform fallback. A reference
that scores both forms or normalizes first-token maxima per label is incorrect.

### 7.4 What the trie is not

It is not a guided-generation engine, KV cache, tokenizer, or model client.
It is also not responsible for deciding which spelling the model prefers:
selection consumes externally supplied first-token scores before the trie runs.
Keeping both pure operations separate makes their contracts directly testable.

## 8. Fixed-context continuation tokenization

V1 compiles exactly two continuation texts per candidate: the candidate itself
and one additional ASCII space followed by it. First-token scores select one
complete path at measurement time. Do not enumerate other segmentations or make
whitespace optional inside a label (§4.3).

### 8.1 Render and freeze the context

For a chat model, render completed messages with the native template. If the
last input message is from the user, open the assistant's final answer position,
including the model-specific reasoning bypass (§9). If the last message is from
the assistant, its content is a final-answer prefill: render the same opening
and bypass, then that exact content, leaving the turn open. An empty final
assistant prefill is equivalent to opening the answer position without a prefill.

Use a supported continuation-of-final-message mechanism or an equivalent tested
model-profile renderer. Do not both open another assistant turn and continue an
existing one. Preserve all prefill characters, including trailing whitespace;
template trimming must not silently change the supplied final assistant content.
Do not run candidate continuations through a chat-template filter that trims them.

Native assistant formatting is model-specific. Preserve role/channel delimiters
and native opening whitespace exactly once. A generation opening need not be
identical to the template's prefix for nonempty assistant content; the profile
must validate both the no-prefill and explicit-prefill cases. Do not invent an
answer cue, move user text into the assistant turn, close the final assistant
turn, or append a second bypass block.

For a template-less base model, use the caller's `prompt` as the context text.
Render and encode once, with the profile's special-token policy, to obtain `C`.
All of `C` is fixed for both forms of every candidate. Neither tokenization nor
scoring may trim, re-encode, shorten, or otherwise rewrite it. No model call or
per-call KV state is needed to prepare this context.

### 8.2 Canonical continuation encoding

The profile provides a deterministic, tokenizer-bound
`encode_continuation(C, surface)` operation. It applies the tokenizer's canonical
segmentation to the **complete** surface with the existing token boundary fixed.
It must not insert BOS/EOS or a dummy start-of-string space absent from that
surface. Use ordinary string encoding when it meets the suffix contract;
otherwise use a tested tokenizer-specific configuration. Do not globally mutate
the tokenizer used to render prompts.

Canonical whole-concatenated-text encoding is not the contract. If `C` ends with
a standalone space token, encoding a bare continuation must not replace that
context token with a combined space-and-word token. Joint-text prefix subtraction
is only a valid optimization when it preserves `C` and reproduces the same paths.

Preparation, shared by preview and measurement:

```text
C = render_and_encode_context(input, reasoning="bypass")
for candidate in candidates:
    forms[candidate] = {}
    for name, surface in (("bare", candidate), ("spaced", " " + candidate)):
        T = encode_continuation(C, surface)
        require T is nonempty
        require decode(C + T) == decode(C) + surface
        forms[candidate][name] = T
validate_potential_paths(forms, END)
```

Decoding retains special tokens and disables whitespace cleanup. Validate
context-plus-path decoding, not concatenated decodes of isolated tokens.
If exact surface preservation fails, raise `TokenBoundaryError` naming the
candidate and form. Do not silently omit a form, change its segmentation to
pass the check, or rewrite context.

The two forms can have different first pieces, token counts, and suffixes.
There is one canonical path per surface, not a search over all tokenizations.
Encoding these alternatives requires no inference. Selecting between them does.

Preflight rejects empty paths, cross-candidate surface/path collisions, and
potential cross-candidate prefix overlaps without a usable, disjoint `END`
(§5.3). A prefix relationship between a single candidate's own two forms is not
an end-vs-continue decision, because only one will be selected. Both forms must
pass exact decoding checks even when a structural representation deduplicates
identical paths.

### 8.3 Build, selection, and reuse timing

`build_credoscope` binds model, candidates, bypass policy, and resolved `END`.
It validates candidate values and overlapping surface forms, without requiring
a sample prompt or predicting future selected paths.

For each measurement:

1. Render `C` and compile/validate both forms (§8.2).
2. Request all distinct first-token ids in one backend readout at `C`.
3. Select one complete path per candidate using §5.1's rule.
4. Build or reuse a pure trie plan for the **selected** paths and `END`.
5. Score the selected root from the existing readout, then query its non-root
   branch nodes. Store the selected paths in `raw.scored_token_ids`.

Preview stops after step 1. It displays the fixed context and all alternatives,
not a selected path or final readout plan. Backend scores are neither available
nor needed to inspect prompt rendering.

A single-entry immutable plan memo is sufficient. Its key is the ordered
candidate-to-selected-path mapping plus `END`, not merely the candidate strings
or both possible forms. Two prompts can compile to the same alternatives but
choose different paths, with different branch points. Always redo score-based
selection per measurement, even on a plan cache hit.

Cache only pure plan data, never context text, first-token scores, logits, or
model/KV state. Input order, eviction, and preview calls cannot affect results.
Reusing a plan does not avoid validating a new context. Later encoding caches
require a tested profile guarantee and profiling evidence.

### 8.4 Walkthrough and regression cases

A non-reasoning Gemma-style answer opening is `"<start_of_turn>model\n"`.
Optional assistant prefills remain literal:

| Final assistant prefill | End of fixed context | Possible completed endings |
|---|---|---|
| None | `"model\n"` | `"model\npositive"`, `"model\n positive"` |
| `"Answer:"` | `"model\nAnswer:"` | `"Answer:positive"`, `"Answer: positive"` |
| `"Answer: "` | `"model\nAnswer: "` | `"Answer: positive"`, `"Answer:  positive"` |
| `"Answer:\n\n"` | `"model\nAnswer:\n\n"` | `"Answer:\n\npositive"`, `"Answer:\n\n positive"` |

The model's first-token scores select one form for each candidate at that context.
Whitespace already in the prefill is neither removed nor moved into the answer.
Added newline/tab/multiple-space surfaces are not considered. Such text can still
be supplied explicitly as prefill or as literal candidate content.

**Different segmentations and changing branches.** Suppose a tokenizer produces:

| Candidate | Bare | Spaced |
|---|---|---|
| positive | `["pos", "itive"]` | `[" positive"]` |
| position | `["pos", "it", "ion"]` | `[" position"]` |

These are illustrative paths, not a claim about a particular model's vocabulary.
The initial query requests the three distinct ids for `"pos"`, `" positive"`,
and `" position"`.

- If both spaced tokens beat `"pos"`, the selected trie has two single-token
  paths. Normalize over those two tokens; no later query is needed.
- If `p(" positive") = 0.40`, `p("pos") = 0.30`, and
  `p(" position") = 0.20`, select spaced positive and bare position. The root
  normalizer is `0.40 + 0.30`; credences are `4/7` and `3/7`.
  After `"pos"`, `"it"` and `"ion"` are forced, with no further readout.
- If `"pos"` beats both spaced tokens, both labels select bare paths.
  The selected root has only `"pos"`, so it is forced with probability 1.
  Query at `C + ["pos"]` to compare `"itive"` with `"it"`; after `"it"`,
  `"ion"` is forced. Do not count `"pos"` twice at the root.

No internal whitespace variants are introduced in any case. Selecting a complete
path preserves whichever suffix that surface's tokenizer encoding specified.

Regression coverage:

- Scalar chat prompt and equivalent user-only messages compile identically.
  Completed few-shot turns remain completed.
- No prefill, empty prefill, punctuation, trailing space, and trailing-newline
  prefills retain exact text. Native controls and bypass occur exactly once.
- Both surfaces decode exactly after the fixed context, including multibyte
  text and tokenizers with dummy-space preprocessing. Unsupported profiles raise
  before inference; context tokens never change.
- Different segmentation lengths and suffixes; no first-token-only substitution.
  Extra surface variants and noncanonical segmentations remain excluded.
- Shared first ids are queried once. Equal finite or `-inf` scores select bare
  deterministically, independently of candidate order.
- The three selected-trie shapes above, including a root selection query when the
  resulting root is forced, and newly absent/present downstream branches.
- Mixed finite/`-inf` scores, all-`-inf` uniform token fallback, and forced steps.
  Zero incoming mass stays zero; token-uniform need not mean label-uniform.
- Preview shows alternatives, not selections. It shares preparation, makes no
  backend calls, and does not mutate later selection or measurements.
- Cold, reused, rebuilt, and preview-interleaved measurements agree within
  backend numerical tolerance. Equal alternatives with different initial scores
  must not accidentally reuse an obsolete selected trie.

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
    ) -> dict[int, float]:
        """Raw logits of every requested token at the next position.

        Values are float32 model logits. `wanted` restricts the returned values,
        not which logits the model is capable of exposing. Every requested id
        must be present; the backend never samples or truncates to top-K.
        """
```

A missing requested id is a backend contract violation: raise
`BackendReadoutError`, not a zero probability. Reject NaN or positive infinity
with the same error. Negative infinity represents zero weight when any allowed
score is finite; an all-`-inf` allowed set triggers uniform token fallback (§5.1).
The backend returns the raw scores; selection, forcing, and fallback belong to
the pure measurement logic, not a backend-specific repair.

The tokenizer adapter exposes:

```python
class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = False,
               clean_up_tokenization_spaces: bool = False) -> str: ...
    def apply_chat_template(self, messages, *, tokenize: bool = False,
                            add_generation_prompt: bool = True,
                            continue_final_message: bool = False,
                            **chat_template_kwargs): ...
    eos_token_id: int | None
    chat_template: str | None
    additional_special_tokens: list[str]
```

The model profile chooses compatible rendering flags; do not open a new assistant
turn and continue a final message simultaneously. The adapter preserves the
special-token policy required by §8. Its tokenizer-bound profile must preserve
an optional final assistant prefill and implement the validated canonical
continuation encoder (§8.2). Neither operation belongs in the backend protocol.

The Credoscope normalizes only over allowed tokens. No full-vocabulary softmax
or transfer of the full vocabulary distribution to the caller is required.
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
The library adds no answer cue. It owns only the two-form continuation policy
(§4.3, §8), keeping native model formatting separate from research instructions,
semantic descriptions, and randomized code meanings.

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
  trie.py            # pure: first-token path selection, §7 trie, readout plans, validation
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
Reusing an immutable selected-trie plan (§8.3) is a separate, pure optimization.
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

## 14. Testing

Three tiers; every test guards a named failure mode — no tests written just to
raise the test-coverage number.

**Tier 1 — pure property tests (Hypothesis + a thin fake backend).** The fake
backend implements `FullLogitsBackend` over a ~10-token vocabulary with declared
logits. Properties:

- **Two-stage parity (the load-bearing one):** first-token selection followed by
  selected-trie scoring equals independent selection plus a naive chain rule.
  Include the §8.4 changing branch shapes and root readout reuse. Normalizing
  candidate maxima before merging shared first tokens must fail parity.
- First-token score ties select the named bare form, not whichever form was
  inserted first. Shared edges count once; cross-label path collisions and empty
  paths raise. Candidate iteration order cannot alter selection or credences
  beyond floating-point tolerance.
- Credences sum to 1 within FP tolerance (per-node masked probabilities over the
  allowed set sum to 1 by construction; log-space accumulation adds rounding);
  an all-`-inf` allowed set yields `1 / len(allowed)` per distinct token. Mixed
  finite/`-inf` scores give zero to `-inf` tokens; finite scores normalize normally.
  Singleton transitions are forced even when a fake backend would score them
  `-inf`, and need no query. Test token-uniform but label-nonuniform outcomes.
- End-vs-continue: `p̃(END) + Σ p̃(continuations) == 1` at the node; the §5.3/§7.1
  `END`-disjointness `ValueError` fires on an overlapping pair; a prefix overlap
  with no resolvable `END` fails potential-path validation before inference.
  All-`-inf` fallback with multiple end ids aggregates their token-uniform mass.
- Log-space robustness at extreme logits, including explicit uniform fallback
  without `-inf - (-inf)`: finite log credences remain finite
  even when exponentiation underflows to zero. JSON serialization distinguishes
  finite logs from exact `-inf` (serialized as `null`).
- Exact ties return all winners in candidate order, with no randomization or
  rounding-induced ties. Caller-seeded tie selection only returns winners;
  fractional weights sum to one and are distinct from the credence vector.
- Duplicate or empty candidates raise as specced.
- Overlapping accepted surface sets, such as `"positive"` and `" positive"`,
  raise before scoring rather than sharing a response between result keys.
- Construction rejects fewer than two candidates; scalar input returns `Measurement`,
  sequence input preserves order and length, and empty sequence input returns `[]`.
- Preview returns `str` or aligned `list[str]` under the same input dispatch;
  preparation errors match measurement, no backend calls occur, and reports show
  alternatives without claiming a selected path or final plan. No automatic
  print/log/file side effects.
- Missing requested backend scores or invalid numeric scores raise
  `BackendReadoutError`, never silently renormalize an incomplete readout.

**Tier 2 — tokenizer-only tests (real tokenizers, no model weights).** Gated on
network/fixture availability, not on hardware — this is the highest-value
regression surface and must run in ordinary CI. Per blessed-registry entry: §8.2
fixed-context canonical continuation encoding and exact decoding of both forms;
the §8.4 prompt/message/prefill cases; bypass inserts and native-disable kwargs;
different whole-form segmentations and potential-path validation. Reject
unsupported rendering/encoding before calling a model. Include assistant
prefills with no trailing whitespace, a trailing space, and trailing newlines.
Preview/measurement preparation parity and faithful report rendering (§4.4) must
cover these same cases, including a user message that itself ends in `"Answer:"`.
With fake first-token scores on those real paths, test cold/warm/rebuilt selected
plans and changing selections even when tokenization is unchanged.

**Tier 3 — opt-in real-model e2e (`CREDENCES_RUN_MLX_TESTS=1`).** Against the
blessed registry's sanity prompts: expected top label and minimum top-label credence,
bypass sanity on at least one `</think>`-family and one Harmony-family model.

Golden fixtures: a hand-built fake-backend scenario per interesting shape —
two single-token alternatives with unequal scores and exact ties;
the three `"positive"`/`"position"` selected-trie shapes (§8.4);
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
- **Supported-model validation:** verify fixed-context continuation encoding of
  both forms with real tokenizer fixtures before declaring support. Validate native
  assistant opening, faithful prefill rendering, and any tokenizer-specific
  suppression of a dummy initial space. A failed fixture calls for an explicit
  profile decision, never context rewriting or silently dropping a form.
- **Broader variants / synonym sets:** v1 accepts only the canonical bare and
  single-ASCII-space forms. Newline/tab/case variants, synonyms, and enumeration
  of all tokenizations are deferred to `POSSIBLE_FUTURE_EXTENSIONS.md` §10.
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
- "On the attribution of confidence to large language models," *Inquiry*, 2025 —
  terminology-in-ML for credence attribution.
