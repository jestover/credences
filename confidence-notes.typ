#set heading(numbering: "1.1")

= Credence Vectors and Confidence Summaries

This note summarizes the conceptual shift behind the planned `credences` package. The central object is not a label and not a single confidence score. The central object is the full probability vector that the language model assigns to a fixed set of admissible labels.

The package should make it easy for researchers to recover that vector, interpret it as the model's credences, and then compute a small number of transparent summaries from it. The confidence summaries are motivated from the credence vector itself. Their relationship to the binary inner-confidence measure in Chen, Didisheim, and Somoza is a useful connection to the existing literature, but it is not the starting point of the framework.

== What We Recover From the LLM

Suppose the researcher supplies a fixed label set

$ L = {l_1, dots, l_k}. $

For a prompt $cal(P)$, the model assigns next-token probability to many possible continuations. We define a constrained procedure over the admissible labels. For notational simplicity, write its prompt-specific credence in label $l_i$ as

$ p_i := p_i (cal(P)) = p(l_i | cal(P)), quad i = 1, dots, k. $

Then the model's credence vector is

$ bold(p)(cal(P)) = (p_1, dots, p_k), quad sum_(i = 1)^k p_i = 1. $

The entry $p_i$ is the model's _credence_ in label $l_i$: its degree of belief under the specified prompt, answer representation, and constraints. This is read from the model's token scores, not a self-reported confidence number or a claim of prompt-independent belief.

For each label, the package tokenizes the bare text and the text prefixed with one ASCII space, then selects the complete path with the higher first-token score; exact ties select the bare form. It builds a trie from those selected paths and normalizes at each decision. A forced transition has probability one; if all allowed token scores at a branch are negative infinity, the procedure uses a uniform token distribution. These are explicit measurement rules, not summation over all spellings or global conditioning on complete answers. Conceptually the output remains simple: one normalized probability per declared label, with semantic code mappings owned by the researcher.

== Why The Vector Matters For Sentiment <sentiment-vectors>

For earnings-call sentiment, the hard label can discard exactly the information we
care about. Consider three labels in the order

$ ("negative", "neutral", "positive"). $

Two prompts might produce

$ A = (0.01, 0.98, 0.01), $

and

$ B = (0.45, 0.54, 0.01). $

Taking the argmax of either credence vector gives the same label: "neutral". This is a label-level summary, not a claim that token-by-token greedy generation must return that label. The vectors have very different meanings. Vector $A$ says the model is overwhelmingly neutral. Vector $B$ says the model is only barely neutral, with nearly all remaining belief leaning negative.

This is the basic value of the credence framework:

$ "discrete label" -> "credence vector" -> "application-specific summary". $

The discrete label is useful, but it is the coarsest representation. The credence vector preserves the model's belief across all admissible labels. A sentiment index or confidence score is then a deliberate summary of that vector.

== Three General Confidence Summaries

The package should expose three first-class summaries that require no assumptions about label ordering or distances. Let

$ p_((1)) >= p_((2)) >= dots >= p_((k)) $

denote the sorted credences.

=== Top-Label Confidence

Define

$ C_"top" = frac(p_((1)) - 1/k, 1 - 1/k). $

This answers:

#quote(block: true)[How strongly does the model favor its most likely label?]

Top-Label Confidence measures how strongly the model favors its most likely label over uniform guessing. The measure is 0 at the uniform distribution and 1 when all credence is on one label.

If we must reduce the credence vector to one categorical label and treat every misclassification as equally costly, the model-implied probability of error is $1 - p_((1))$. `top_label_confidence` rescales the reduction in that error probability so that 0 means no improvement over uniform guessing and 1 means all credence is on one label. In decision-theoretic terms, this is reverse normalized Bayes risk under zero-one loss.

This makes $C_"top"$ useful when the question is not “how spread out is the whole distribution?” but “how strong is the model’s best single-label decision?”

In the binary case,

$ C_"top" = 2 abs(p - 1/2). $

So top-label confidence is exactly twice the Chen, Didisheim, and Somoza binary inner-confidence measure. Dividing by 2 gives the paper's scale, while the normalized version puts the measure on a common 0-to-1 scale across label sets with different numbers of labels.

=== Entropy Confidence <ent>

Define Shannon entropy as

$ H(bold(p)) = - sum_(i = 1)^k p_i log p_i, $

and define

$ C_H = 1 - H(bold(p))/(log k). $

This answers:

#quote(block: true)[How concentrated is the model's full credence distribution?]

Unlike top-label confidence, entropy confidence uses the entire vector. It distinguishes a case where the losing mass is spread across many plausible labels from a case where uncertainty is concentrated in one close alternative. Because $log k$ is equal to the max possible entropy that occurs when the credences follow a uniform distribution $C_H$ also lives on a scale from 0 to 1.

This is the direct multiclass version of the reverse-entropy idea emphasized in the inner-confidence paper. In binary problems, the paper's inner-confidence measure is monotone in reverse entropy. For $k > 2$, top-label confidence and entropy confidence can disagree because they answer different questions.

For example,

$ (0.60, 0.20, 0.20) $

has a stronger top label than

$ (0.55, 0.45, 0), $

but the second distribution has lower entropy because almost all uncertainty is a two-label contest. Top-label confidence ranks the first as more confident in the top label. Entropy confidence ranks the second as more concentrated overall. Both measures are coherent.

=== Margin Confidence

Define

$ C_"margin" = p_((1)) - p_((2)). $

This answers:

#quote(block: true)[How far ahead is the preferred label from its nearest competitor?]

Margin confidence is especially useful when the practical question is not how much total mass sits outside the top label, but whether the top label is clearly ahead of the closest alternative.

In the binary case,

$ C_"margin" = 2 abs(p - 1/2), $

so it also equals twice the Chen, Didisheim, and Somoza binary inner-confidence measure. For more than two labels, it becomes a distinct summary because it ignores alternatives below the runner-up.

== Why These Are Not One Measure

The three summaries are intentionally separate:

- `top_label_confidence` measures confidence in the best single-label decision.
- `entropy_confidence` measures concentration of the full belief distribution.
- `margin_confidence` measures decisiveness relative to the closest competitor.

In binary problems, these ideas collapse into nearly the same object because the probability simplex has only one degree of freedom. With three or more labels, they separate.

The package should therefore report the vector and these named summaries rather than trying to make one scalar carry every meaning of confidence.

== Ordered Labels Require A Geometry

For ordered labels, like our sentiment labels, we may also want a directional measure. That requires an additional researcher-supplied geometry. For example, assign

$ "negative" = -1, quad "neutral" = 0, quad "positive" = 1. $

Then expected sentiment is

$ S = sum_i p_i x_i = p_"pos" - p_"neg". $

For the two motivating vectors in @sentiment-vectors,

$ S(A) = 0 $

while

$ S(B) = 0.01 - 0.45 = -0.44. $

This captures the fact that even though both prompts are argmax-neutral, the second leans substantially negative.

Once we assign numeric scores $x_i$, we can also compute ordinal dispersion:

$ sigma = sqrt(sum_i p_i (x_i - mu)^2), quad mu = sum_i p_i x_i. $

This should be described as sentiment or ordinal dispersion, not as another generic confidence measure. It combines two ingredients:

$ "model credences" + "researcher's geometry of label distances". $

The equal spacing in $(-1, 0, 1)$, or in $(-1, -0.5, 0, 0.5, 1)$ if we also want labels for "very negative" and "very positive", is a modeling assumption. It is often a reasonable benchmark for sentiment, but it is not learned from the LLM probabilities alone.

== Using The Full Vector Empirically

The finance application can compare three increasingly rich representations:

1. the discrete argmax label;
2. expected sentiment from a chosen geometry;
3. the full credence vector.

For three labels, expected sentiment is

$ S_i = p_(i,"pos") - p_(i,"neg"). $

A return regression

$ R_i = alpha + beta S_i + epsilon_i $

imposes the restriction that positive and negative credences have equal and opposite effects on returns

An unrestricted credence-vector regression, omitting one probability because the probabilities sum to one, is

$ R_i = alpha + beta_"pos" p_(i,"pos") + beta_"neg" p_(i,"neg") + epsilon_i. $

Then expected sentiment corresponds to the testable restriction

$ beta_"pos" = - beta_"neg". $

With five labels, the same logic asks whether the chosen scale $(-1, -0.5, 0, 0.5, 1)$ is empirically adequate, or whether very negative, negative, neutral, positive, and very positive credences have asymmetric or nonlinear relationships with returns.

This makes the geometry useful without pretending it is automatic. Expected sentiment is a parsimonious projection. The full vector is the unrestricted benchmark.

== Contributions

The software package contribution is:

#quote(block: true)[
  Discrete LLM outputs discard information contained in the model's credence distribution. The package recovers that distribution for a researcher defined set of labels and provides transparent, first-principles summaries of top-label confidence, distributional concentration, and nearest-competitor margin.
]

The sentiment analysis application then builds on that:

#quote(block: true)[
  For ordered financial sentiment, the credence vector can be projected onto a sentiment scale, or used directly to test whether that projection captures the economically relevant information.
]

That separation should make both projects cleaner. The package paper extends the inner-confidence literature to multiclass forced-choice credences by starting with the full vector. The sentiment paper uses the vector to recover variation that is lost when earnings-call text is collapsed to an overwhelmingly neutral hard label.
