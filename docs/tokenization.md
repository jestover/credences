# How answers become tokens

`credences` turns a language model's next-token probabilities into a
probability for each answer you list. This page explains what happens between
your answer strings and the numbers you get back. You do not need it to use the
package. Read it if you want to know exactly what a credence measures, or why
`"yes"` and `"Yes"` count as different answers.

## How a language model writes text

**Tokens.** A language model does not read or write letters or whole words. It
works with *tokens*: pieces of text from a fixed vocabulary, usually tens of
thousands to a few hundred thousand entries, each with an integer ID. The
vocabulary is built before training by a procedure that resembles compression:
strings that appear often in the training text get their own token. A common
word is usually one token, and a rare word is split into several pieces. The
sentence `Revenue beat expectations.` might become five tokens: `Rev`, `enue`,
` beat`, ` expectations`, and `.`. Every model has its own vocabulary, so this
split is only an illustration. A *tokenizer* converts text to token IDs and back.

**One token at a time.** Given the tokens so far, the model computes a score,
called a *logit*, for every token in its vocabulary. A softmax turns the scores
into probabilities:

```text
p(token i | text so far) = exp(z_i) / sum over all tokens j of exp(z_j)
```

If you know discrete-choice models, this is a multinomial logit with one
alternative per vocabulary entry, where the network computes the utilities `z`
from everything that came before. The model picks a token, appends it to the
text, and repeats. So the probability of a reply several tokens long is a
product of conditional probabilities, one per token. `credences` never lets
the model write anything; it only reads these probability distributions.

## Why "yes" and " yes" are different tokens

Most vocabularies attach the space between two words to the word that follows.
The text `Answer: yes` is typically encoded as `Answer`, `:`, ` yes`, and
` yes`, with its leading space, is a different token from `yes`. It has a
different ID and its own probability. The same goes for `Yes`, ` Yes`, and
`YES`.

Which form the model uses depends on what comes just before the answer, as it
would in ordinary writing. Most chat formats start the model's reply on a
fresh line or right after a special marker, where a word normally has no space
in front, so the model tends to write `yes`. After `Answer:`, a space normally
follows the colon, so the model tends to write ` yes`. Some chat formats, such
as Mistral's, put a space before every reply.

Suppose that after `Answer:` a model gives these probabilities to its next
token:

| Next token | Probability |
|---|---|
| ` yes` | 0.60 |
| ` no` | 0.25 |
| `yes` | 0.03 |
| `no` | 0.02 |
| anything else (`The`, a line break, ...) | 0.10 |

Reading only `yes` and `no` would give yes a share of 0.03 / 0.05 = 0.60, based
on 5% of the model's probability and ignoring how it actually writes its
answer. Reading only ` yes` and ` no` works here but would fail after a line
break. So `credences` reads both forms and adds them.

## What credences scores

For each answer you list, `credences` scores two *spellings*:

- the **bare** spelling, the answer as written: `yes`;
- the **spaced** spelling, the answer after one space: ` yes`.

An answer's credence is the total probability of its two spellings. When the
model clearly prefers one, the total is close to that one alone. When it is
undecided about the space, both count, and the result changes smoothly with
the model's scores instead of jumping from one spelling to the other.

The rules around this:

- **Answers are trimmed at the ends.** `" yes "` is scored as `yes`, because
  the two spellings already cover whether the answer starts with a space.
  Spaces inside an answer, as in `very positive`, are kept. Results are keyed by
  your answers exactly as you wrote them.
- **Nothing else is added.** Capitalized forms, a line break or quotation mark
  before the answer, and replies such as `The answer is yes` are different
  outputs and are not counted. List your answers in the form your prompt asks
  for. If `Yes` and `yes` should mean the same thing in your analysis, list
  both and add their credences afterward.
- **Each spelling is tokenized together with your prompt.** A tokenizer can
  split text differently depending on its neighbors. If your prompt ends with
  `Answer: ` (with a trailing space), the full text `Answer: yes` is encoded
  with that space inside the ` yes` token. Encoding the prompt and the answer
  separately would give `Answer`, `:`, ` `, `yes`, a sequence the model almost
  never saw in training. So `credences` encodes the prompt and each spelling as
  one text. Scoring starts right after the tokens that the prompt and every
  spelling share, so the model sees exactly what it would see if the whole text
  appeared in its input.
- **Your prompt is never changed.** The prompt, and any text you put at the
  start of the model's reply (a *prefill*, such as `Answer:`), are used exactly
  as written, including trailing spaces and line breaks. Only the point where
  scoring starts can move, as in the trailing-space case above.
- **One tokenization per spelling.** Each spelling is scored in the tokens the
  tokenizer gives it. Other token sequences that spell the same text, such as
  `y`, `e`, `s`, are not added: tokenizers never produce them, and research on
  this finds that the standard tokenization usually carries almost all of a
  text's probability (Geh et al., 2024).

## From token probabilities to credences

**Restricted to your answers.** At each step, `credences` keeps only the
tokens that continue one of your answers' spellings and rescales their
probabilities to sum to 1. With one-token answers, this is ordinary
conditional probability: each answer's probability given that the model writes
one of them. In the table above, the four allowed tokens have total probability
0.90, so

```text
credence(yes) = (0.60 + 0.03) / 0.90 = 0.70
credence(no)  = (0.25 + 0.02) / 0.90 = 0.30
```

**Answers longer than one token.** The restriction applies at every step. A
credence is therefore the probability that a writer who must stay within your
answers, choosing one token at a time with the model's probabilities, produces
that answer. This differs slightly from computing each complete answer's
unrestricted probability and rescaling at the end: once the model has started
an answer, its next choice is rescaled among the allowed continuations.

Answers that start with the same token share that step. Suppose `positive` is
encoded as `pos`, `itive` and `position` as `pos`, `it`, `ion`. The model's
probability of `pos` is counted once, then divided between `itive` and `it`
according to the model's next-token probabilities. After `pos`, `it`, only
`ion` is allowed, so that step counts as certain.

When one answer is the start of another, as `up` is of `upward`, the model also
decides after `up` whether to stop. Its probability of stopping is read from
its end-of-answer tokens, such as the token a chat model writes to end its
reply.

The credences always sum to 1 across your answers, without rescaling at the
end. They describe how the model divides its probability among the answers you
gave, for that prompt. They are not guaranteed to be calibrated.

## Coverage

Each measurement also reports *coverage*: the model's unrestricted probability
that its first token starts one of your answers' spellings. In the table above,
coverage is 0.90. A first token can still continue into something else, so
coverage is an upper bound on how likely the model was to answer in your format
without the restriction.

Coverage never changes credences, and low coverage does not mean worse
credences. Here are two models answering the same yes/no question:

| Model | Next-token probabilities | Coverage | Credences |
|---|---|---|---|
| 1 | `The` 0.90, `yes` 0.07, `no` 0.02 | 0.09 | yes 0.78, no 0.22 |
| 2 | `yes` 0.70, `no` 0.20, `The` 0.01 | 0.90 | yes 0.78, no 0.22 |

Model 1 likes to open with "The"; model 2 answers directly. Their credences are
identical, and either could be the more accurate one. Whether coverage relates
to accuracy is an empirical question that depends on the model. `credences`
reports it for inspection and never filters or reweights results by it.

## What is implemented so far

The scoring steps are implemented: adding the two spellings, restricting each
step to your answers, sharing common starts, reading the stop decision, and
coverage. Trimming answers, encoding them with your prompt, and handling
prefills arrive with model loading. Until then, the lower-level functions take
token IDs you supply, as the [walkthrough](pure-core.md) shows.

## Reference

R. L. Geh, H. Zhang, K. Ahmed, B. Wang, and G. Van den Broeck,
[*Where is the signal in tokenization space?*](https://arxiv.org/abs/2408.08541),
EMNLP 2024.
