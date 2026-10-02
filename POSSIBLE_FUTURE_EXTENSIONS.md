# Possible Future Extensions

This note preserves ideas that are useful, but should not be part of the minimal
`credences` core. The core package should recover the credence vector and provide
the first summaries that follow directly from it. The extensions below require
extra researcher-supplied structure or add responsibilities beyond measurement,
such as an ordering, loss function, distance or similarity matrix, embedding
model, or presentation design.

Each entry states the use case, proposed behavior, and main caveats so it can be
evaluated independently later.

## 1. Ordinal Boundary Confidence

**Use when:** labels have a meaningful order, such as
`very negative < negative < neutral < positive < very positive`, but we do not
want to assume a fully cardinal scale.

The idea is to turn an ordered `k`-label problem into `k - 1` binary questions.
For ordered labels

```text
y_1 < y_2 < ... < y_k
```

define the cumulative probability below each boundary:

```text
F_m = Pr(Y <= y_m) = sum_{i=1}^m p_i,    m = 1, ..., k - 1.
```

Each boundary asks a binary question:

```text
{y_1, ..., y_m} versus {y_{m+1}, ..., y_k}.
```

The boundary-specific confidence is the binary inner-confidence idea applied to
that cumulative split, usually normalized to `[0, 1]`:

```text
b_m = 2 |F_m - 1/2|.
```

This is high when the model is clearly on one side of the boundary and low when
the model is split evenly across it.

An aggregate ordinal confidence can average the boundary values:

```text
C_ord = (sum_{m=1}^{k-1} w_m b_m) / (sum_{m=1}^{k-1} w_m),
```

where `w_m > 0` are optional boundary weights. Equal weights mean that crossing
each adjacent label boundary matters equally.

Why this is useful:

- It distinguishes probability split across adjacent labels from probability split
  across distant labels.
- It can say that `(0.5, 0.5, 0)` is more ordinally concentrated than
  `(0.5, 0, 0.5)` for labels `(negative, neutral, positive)`, even though both
  distributions have the same top-label confidence and the same Shannon entropy.
- The vector `(b_1, ..., b_{k-1})` is often more informative than the aggregate
  because it shows which boundary is uncertain.

Decision-theoretic connection:

If adjacent label gaps have weights `w_m`, the minimum expected absolute-error
loss on the ordered scale can be written as

```text
R_abs^*(p) = sum_{m=1}^{k-1} w_m min(F_m, 1 - F_m).
```

Since

```text
2 |F_m - 1/2| = 1 - 2 min(F_m, 1 - F_m),
```

ordinal boundary confidence is a reverse-normalized version of that minimum
model-implied ordinal risk:

```text
C_ord = 1 - 2 R_abs^*(p) / (sum_m w_m).
```

Caveat:

The label order is extra structure, and any weights are a modeling choice. Equal
boundary weights may be a reasonable default for sentiment labels, but they are
not learned from the LLM credences alone.

## 2. Decision Confidence Under A User-Supplied Loss Matrix

**Use when:** different mistakes have different costs.

Top-label confidence assumes every incorrect categorical label is equally costly.
That is often too simple. For example, in a finance application, calling a very
negative disclosure "positive" may be more costly than calling it "negative" or
"neutral"; false positives and false negatives may also have asymmetric costs.

Let `L(a, j)` be the loss from choosing action or predicted label `a` when the
true label is `j`. Given credences `p_j`, the model-implied risk of action `a` is

```text
R_L(a; p) = sum_j p_j L(a, j).
```

The best achievable model-implied risk is

```text
R_L^*(p) = min_a R_L(a; p).
```

A decision-confidence score can reverse and normalize this risk:

```text
C_L(p) = 1 - R_L^*(p) / R_{L,max},
```

where `R_{L,max}` is the largest possible minimum risk over the probability
simplex:

```text
R_{L,max} = max_{p in Delta_k} R_L^*(p).
```

In practice, the normalization may need to be chosen carefully. Alternatives
include normalizing by the worst-case Bayes risk, the risk under a uniform
distribution, or a domain-specific benchmark.

Why this is useful:

- It generalizes top-label confidence.
- It allows asymmetric costs.
- It can make the best decision differ from the most probable label when losses
  are asymmetric.

Special case:

For zero-one loss,

```text
L(a, j) = 0 if a = j, and 1 otherwise,
```

the best action is the top label, `R_L^*(p) = 1 - p_(1)`, and the normalized
reverse risk becomes the top-label confidence measure.

Caveat:

A loss matrix is a researcher's modeling choice. The package can support this
later, but the core package should not invent costs for users.

## 3. Pairwise Dispersion From A Distance Matrix

**Use when:** the question is "how far apart are the plausible labels?" rather
than "how risky is the best single decision?"

Suppose the researcher supplies a distance matrix `D`, where `d_ij` measures how
far apart labels `i` and `j` are. A natural dispersion measure is

```text
Q_D(p) = sum_i sum_j p_i p_j d_ij.
```

This is the expected distance between two independent labels drawn from the
model's credence distribution:

```text
Q_D(p) = E[d(Y, Y')].
```

Why this is useful:

- It measures spread over a structured label space.
- It does not require choosing a single predicted label.
- It separates "two nearby labels are plausible" from "two distant labels are
  plausible."

Special cases:

If labels have numeric scores `x_i` and the distance is squared distance,

```text
d_ij = (x_i - x_j)^2,
```

then

```text
Q_D(p) = 2 Var(X).
```

So ordinary variance is a special case of pairwise dispersion.

If labels are nominal and

```text
d_ij = 1{i != j},
```

then

```text
Q_D(p) = 1 - sum_i p_i^2,
```

which is the Gini or Simpson impurity.

Caveat:

The distance matrix must mean something for the task. For sentiment, an ordinal
distance matrix may be natural. For unrelated categories, distances can easily
become arbitrary.

## 4. Similarity-Sensitive Entropy

**Use when:** labels are not identical, but some labels are more similar than
others, and splitting mass across similar labels should count as less uncertainty
than splitting mass across very different labels.

Ordinary Shannon entropy treats labels as unrelated atoms. A distribution
`(0.5, 0.5, 0)` has the same entropy no matter which two labels receive the mass.
That is appropriate for nominal labels with no structure, but not always for
semantic or ordered labels.

A similarity-sensitive entropy starts with a similarity matrix `Z`, where
`Z_ij` measures how similar label `i` is to label `j`. A common form is

```text
H_Z(p) = - sum_i p_i log((Z p)_i),
```

where

```text
(Z p)_i = sum_j Z_ij p_j.
```

If `Z` is the identity matrix, then `(Z p)_i = p_i`, and this reduces to ordinary
Shannon entropy:

```text
H_I(p) = - sum_i p_i log p_i.
```

If two labels are similar, probability split between them contributes less
effective uncertainty than probability split between dissimilar labels.

Why this is useful:

- It keeps an entropy-like interpretation while allowing label relationships.
- It can distinguish adjacent sentiment ambiguity from opposite-sentiment
  ambiguity.
- It may be useful for label sets with semantic overlap but no simple ordering.

Caveats:

- The similarity matrix is extra structure and may be controversial.
- The maximum entropy depends on `Z`, so normalization is less automatic than
  ordinary `1 - H(p) / log(k)`.
- Similarity-sensitive entropy is elegant but probably too complex for the first
  package version.

## 5. Embedding-Based Label Distances Or Similarities

**Use when:** labels have semantic relationships, but the researcher does not have
a natural hand-coded ordering, loss matrix, or distance matrix.

The idea is to embed each label, label definition, or templated label description
into a vector space. For example, instead of embedding only the phrase
`very negative`, embed a fuller description such as:

```text
The passage expresses strongly negative financial sentiment.
```

Let `e_i` be the embedding for label `i`. Distances could be defined by Euclidean
distance,

```text
d_ij = ||e_i - e_j||_2,
```

or by angular / cosine distance. Similarities could be generated from distances,
for example:

```text
Z_ij = exp(-d_ij / tau),
```

where `tau` controls how quickly similarity decays with distance.

The resulting distance or similarity matrix could then feed into:

- pairwise dispersion;
- decision confidence under a loss matrix;
- similarity-sensitive entropy.

Why this is useful:

- It can handle multi-token labels naturally because the whole phrase or
  definition is embedded at once.
- It can support structured categories that are not simply ordinal.
- It may help when labels have semantic relationships that are hard to encode by
  hand.

Caveats:

- Embeddings measure linguistic or semantic relatedness, not necessarily
  task-specific error cost.
- For sentiment, "positive" and "negative" may be semantically related because
  both are sentiment words, even though they are opposites for the task.
- The embedding model, prompt template, distance metric, and any kernel parameter
  must be fixed and versioned.
- The resulting matrix should be inspected before use; it should not be treated as
  an automatic ground truth.

## 6. Entropy-Equivalent Top Probability Index

**Use when:** we want a scalar that preserves reverse-entropy ordering exactly,
but is expressed on a probability-like scale.

For `k` labels, define a canonical distribution with one dominant label and the
remaining probability spread equally:

```text
(q, (1 - q)/(k - 1), ..., (1 - q)/(k - 1)),
    q in [1/k, 1].
```

Its entropy is

```text
Phi_k(q) = -q log q - (1 - q) log((1 - q)/(k - 1)).
```

For any credence vector `p`, find the unique `q_E` such that

```text
Phi_k(q_E) = H(p).
```

Then define

```text
C_E(p) = q_E - 1/k,
```

or normalize it to `[0, 1]` by dividing by `1 - 1/k`.

Interpretation:

`q_E` is the top-label probability that a canonical "one label versus equally
likely alternatives" distribution would need in order to have the same entropy as
the actual credence vector.

Why this is useful:

- It preserves reverse Shannon entropy ordering exactly.
- In the binary case, it reduces numerically to the Chen, Didisheim, and Somoza
  inner-confidence measure on their scale.
- It can help readers who want an entropy-based measure expressed as a
  probability-like dominant-label index.

Caveat:

This is harder to explain than normalized reverse entropy and requires a
one-dimensional numerical inversion. For the core package, plain
`entropy_confidence = 1 - H(p) / log(k)` is likely clearer.

## 7. Standalone Prompt Renderer And Presentation Randomization

**Use when:** researchers want reproducible prompt construction or need to test
whether option order and code assignment affect measured credences.

This should be a standalone helper, not part of the Credoscope. The Credoscope
continues to know its candidate strings and two-form continuation policy, not
their semantic meanings. A renderer could return a record such as:

```python
@dataclass(frozen=True)
class RenderedPrompt:
    id: str
    prompt: str | ChatSpec
    candidates: tuple[str, ...]
    candidate_to_label: dict[str, str]
    condition: dict[str, object]
```

For verbal candidates, `candidate_to_label` may be the identity mapping. For a
coded prompt it records the meaning of each code for that input, for example
`{"A": "neutral", "B": "positive", "C": "negative"}`. The mapping must be a
bijection covering every candidate. The stable `id`, mapping, condition metadata,
raw candidate credences, and remapped semantic credences should be persisted
together; distributed pipelines should not rely only on list position.

Potential presentation controls include:

- permuting the order in which semantic labels or answer options appear;
- assigning a fixed code set such as `A/B/C` to semantic labels under balanced
  cyclic permutations;
- crossing option order with code assignment when interactions matter;
- recording seeds and condition identifiers so every prompt is reproducible;
- optionally comparing a small number of deliberately chosen code schemes.

Balanced reassignment of a fixed code set controls label-to-code confounding more
efficiently than choosing arbitrary new symbols for every prompt. Fully random
code sets also vary tokenization, token frequency, and learned symbol associations,
which adds variance and makes scopes harder to reuse. Different code schemes are
therefore best treated as explicit experimental conditions, not incidental
per-prompt randomness.

The helper could group rendered records by identical `candidates` tuples so one
Credoscope is reused for each candidate set. It may eventually belong in
`credences`, but a separate package or analysis-specific implementation may be a
better home if prompt design becomes domain-specific.

This task-level renderer is distinct from the core native chat rendering and
two-form continuation policy in SPEC §4.3 and §8. It can supply an assistant
prefill through `messages=`, but semantic mappings and presentation experiments
remain outside the Credoscope.

## 8. Measurement After Generated Reasoning

**Use when:** researchers want to compare bypass measurements with answer
distributions after a model has generated a thought trace. This is deferred;
v1 accepts only `reasoning="bypass"` and has no generation or token-cap machinery.

The quantities must remain distinct:

- Bypass measures `P(answer | prompt, bypass context)`.
- A possible `reasoning="allow"` measures `P(answer | prompt, one realized trace)`.
- Averaging over sampled traces would target a marginal under a specified trace
  generation policy. One greedy trace is not that marginal.

Reasoning may commit the model to a decision before the answer readout, leaving
a sharply concentrated distribution. It may also improve a task's answer
distribution. Conversely, bypass may be off-distribution for a model trained to
always think. These are research questions, not guarantees about either mode.

Useful implementation constraints to preserve for later:

- Resolve a model-specific stopping profile. Detecting a closing thinking marker
  alone may not reach the answer position: newlines, channel markers, and other
  answer-prefix tokens can still be required.
- Keep generated token ids as the authoritative context. Decoding a short tail
  and encoding it again is not guaranteed to preserve tokenization. Do not
  silently replace generated tokens to manufacture a convenient answer boundary.
- Bound generation, but treat exhaustion as failure to reach the intended
  answer boundary. Appending a synthetic closing marker changes the context;
  do not report that as an ordinary completed-reasoning measurement.
- V1 places a caller's final assistant prefill after reasoning bypass. A future
  generated-reasoning mode must separately specify post-trace ordering: generate
  the trace first, then append the prefill, or use another explicit protocol.
  Do not treat v1 prefill support as permission to rewrite generated context.
- Any future backend generation operation needs an explicit capability and stop
  contract. Do not add a placeholder generation method to v1 protocols.
- Sampled-trace experiments should use caller-controlled randomness, define the
  handling of failed/truncated traces, and keep calls independent. Researchers
  own trace storage and experiment metadata; no core provenance class is implied.

Validate the core two-form continuation and fixed-context implementation before designing
this extension. It should reuse that contract at the post-reasoning answer
position. Add result fields only for information the extension actually needs;
v1 has no redundant per-measurement reasoning-mode field.

### 8.1 Jeeves: relevant evidence and limits

Source review: [PostHog/Jeeves](https://github.com/PostHog/jeeves), commit
`6151619c14fcffb2406830f09fd4f09fdd22200e`, reviewed 2026-09-30. These are
observations from its source and reported results, not independently reproduced
benchmarks.

- **Its probabilities come from a trained classifier, not answer-token logits.**
  Jeeves trains Qwen3.5-9B with LoRA and a pointer head. After the thought trace,
  its protocol presents the options again and inserts a decision marker; the
  head scores option representations against the decision representation.
  Softmax, with a fitted temperature, produces the vector. This avoids measuring
  the spelling of an answer already stated in the trace, but still conditions
  on that trace. It does not establish that an unmodified model's token readout
  will preserve uncertainty. Adopting the head would mean training a different
  measurement instrument, not adding a reasoning switch to the Credoscope.
  [Head implementation](https://github.com/PostHog/jeeves/blob/6151619c14fcffb2406830f09fd4f09fdd22200e/model/head.py),
  [input construction](https://github.com/PostHog/jeeves/blob/6151619c14fcffb2406830f09fd4f09fdd22200e/loader/dataloader.py).
- **It explicitly acknowledges over-sharpening, but not our exact mechanism.**
  The README says training stopped at step 402, warning that beyond that point
  "the head over-sharpens on the saturated RL pool." This is a warning about
  training dynamics, not a demonstration of answer-token saturation after a
  trace precommits to a label. It reports test accuracy of 0.840 with thinking
  versus 0.804 without it. Better decisions and informative uncertainty are
  separate properties; that comparison alone cannot establish the latter.
  [Training and results](https://github.com/PostHog/jeeves/blob/6151619c14fcffb2406830f09fd4f09fdd22200e/README.md).
- **Its diagnostics are worth borrowing.** The data generator creates paired
  intact and "unknowable" questions by removing decisive evidence, assigning
  uniform targets to the latter. Evaluation tracks mean maximum probability
  and the share at or above 0.9, including after thinking. Its JevBench evaluator
  compares thinking with no-thinking and reports top-label calibration error,
  but saves only the winning probability, not the whole vector. These are useful
  starting points, not a published analysis of our precommitment failure mode.
  [Paired data](https://github.com/PostHog/jeeves/blob/6151619c14fcffb2406830f09fd4f09fdd22200e/prep/night2.py#L82),
  [confidence diagnostics](https://github.com/PostHog/jeeves/blob/6151619c14fcffb2406830f09fd4f09fdd22200e/predictor.py#L158),
  [JevBench evaluation](https://github.com/PostHog/jeeves/blob/6151619c14fcffb2406830f09fd4f09fdd22200e/jevbench.py#L57).
- **Calibration is an additional learned transformation.** Its calibration CLI
  fits one temperature on no-thinking development rows, minimizing negative
  log likelihood; the same head temperature is used for thinking evaluations.
  Our inference: transfer between modes needs testing, not assuming. Flattening
  a vector does not recover disagreement across unrealized reasoning traces.
  Any future downstream calibration should remain distinct from raw Credoscope
  measurements and be validated on held-out data for the relevant mode.
  [Calibration entry point](https://github.com/PostHog/jeeves/blob/6151619c14fcffb2406830f09fd4f09fdd22200e/calibrate.py#L14),
  [temperature fitting](https://github.com/PostHog/jeeves/blob/6151619c14fcffb2406830f09fd4f09fdd22200e/predictor.py#L131).
- **Truncation is a substantive policy choice.** Its engine appends the closing
  marker when a trace reaches its cap without finishing, then scores the options;
  a `closed` flag distinguishes natural completion. Preserve that distinction,
  not the forced-close default: for the prospective Credoscope contract above,
  budget exhaustion remains failure to reach the intended boundary unless a
  separately named truncation policy is explicitly designed and evaluated.
  [Inference boundary handling](https://github.com/PostHog/jeeves/blob/6151619c14fcffb2406830f09fd4f09fdd22200e/inference/engine.py#L431).

### 8.2 Experiments before implementing a reasoning mode

These are proposed Credoscope experiments, not conclusions established by Jeeves:

- Compare bypass with one completed trace on the same inputs, candidates, and
  selected-form constrained readout policy. Save full vectors in the research
  harness. Measure accuracy separately from entropy, maximum credence, saturation
  rates (for example, above 0.99 and 0.999), confident errors, Brier score, negative
  log likelihood, and calibration. Concentration is not itself a defect when
  justified by the task; the question is whether it tracks evidence and errors.
- Include ambiguous or evidence-removed examples with answerable controls.
  Inspect whether traces explicitly choose a label before readout, and compare
  naturally completed traces across token budgets. Do not assume missing evidence
  implies uniform beliefs in arbitrary tasks; justify target distributions from
  the experimental construction. Failed/truncated traces must not disappear from
  reported completion rates.
- Compare a single trace with an average over independently sampled traces:
  `q_bar = mean_r q(answer | prompt, trace_r, readout policy)`. Ten nearly one-hot
  traces that split six/four between labels can yield a roughly 0.6/0.4 average;
  ten agreeing traces will not. Compare entropy of the average with average
  per-trace entropy to distinguish between-trace disagreement from uncertainty
  within a trace. This targets the specified sampling policy, not an automatic
  recovery of bypass credences or a model's uniquely defined "true belief."
  Keep trace-sampling temperature separate from any probability calibration.
- Keep trace generation, boundary validation, and constrained readout separable.
  Preserve generated token ids and test the post-trace context explicitly.
  Evaluate the simplest correct implementation before adding adaptive reasoning
  gates, speculative decoding, or other runtime optimizations. Generation and
  readout parity, not speed alone, must govern any later optimization.

None of these notes adds a v1 mode, result field, calibration parameter, or
training requirement. The core remains `reasoning="bypass"` only.

## 9. Restricted Probability APIs

**Use when:** a concrete research question requires a model accessible only through
an API that does not satisfy the full-logits contract. This is not a planned
release milestone, and v1 has no placeholder protocol or provider adapter.

Such an extension must establish its own capability limits without weakening
the core measurement contract:

- Can the caller specify the exact context and continue at the intended answer
  position? Is a compatible tokenizer available?
- Are scores available for every required token, including low-probability ones,
  or only a truncated top-K set? A missing score is not a known zero.
- Can multi-token candidate prefixes be scored, or only the first answer token?
- If multiple forms are accepted, every required form needs a score. One observed
  spelling per candidate does not establish complete coverage.
- Any approximate or incomplete result would need an explicit, separately
  justified contract. Do not silently return the core `Measurement` after filling
  missing scores with zeros or renormalizing a partial set.

Provider capabilities must be verified when this extension is considered; do not
base the core architecture on a snapshot of hosted API behavior. A server that
does expose arbitrary requested logits at supplied token contexts can instead
implement the ordinary full-logits backend contract, regardless of its location.

## 10. Broader Accepted Answer Forms

**Use when:** researchers explicitly want forms beyond the core bare and
single-ASCII-space variants, such as leading newlines, `"Positive"`, or synonyms;
or want to include alternative tokenizations of the same surface string.

V1 already compiles `candidate` and `" " + candidate`, with one canonical
continuation encoding per form under one fixed context. First-token scores select
one complete path per candidate before constrained scoring. It does not enumerate
paths such as `["p", "os", "itive"]` or add newline prefixes unless they are
literal parts of a declared candidate. Broader forms change the allowed output
language, not merely parsing. Their excluded probability need not be negligible.

Any broader extension must specify whether it still selects one form or instead
sums probabilities over all admitted forms. Those are different measurement
definitions; summing would require one union trie and log-space aggregation by
candidate, not averaging independently normalized distributions. Retain disjoint
path ownership, shared-prefix accounting, fixed context, and explicit stopping
rules. Update raw-path reporting and preview semantics to match the chosen policy.

A grammar/vocabulary compiler could enumerate alternatives without users listing
them, but summing all paths may require many more model evaluations than selecting
one representation. Bound whitespace alternatives and establish a concrete
research benefit before expanding the core contract. Vocabulary tokens may combine
whitespace with text or represent whitespace alone; a first-piece regex search
does not capture every whole-label canonical encoding.

Spelling equivalence is distinct from the caller's mapping of codes to semantic
labels. This extension would not give the Credoscope ownership of those meanings,
nor change masked scoring into global conditioning on valid complete answers.
