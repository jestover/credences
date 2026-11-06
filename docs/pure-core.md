# Small examples of the pure measurement core

These examples use **invented token IDs and logits**, not a real tokenizer or
model. They show the lower-level building blocks that the planned model-backed
API will run for you; that API is not implemented yet. Run the snippets on this
page in order.

`>>>` and `...` mark interactive Python input; unprefixed lines are expected
output. To turn a snippet into a script, remove the prompts and output lines.

Outputs are rounded here only for readability; the library itself never rounds.
`uv run pytest` runs every example on this page, so they stay in step with the code.
To try them yourself, run `uv sync --group dev` in a clone of this repository,
then start `uv run python`.

## 1. From two spellings per answer to a scored trie

### Two spellings

Each answer is scored as written and after one space, and its credence is the
**total** probability of the two. [How answers become tokens](tokenization.md)
explains why. The two spellings can differ in both length and token IDs, not
just their first piece: a hypothetical tokenizer might split `"positive"` into
`"pos"`, `"it"`, and `"ive"` with IDs `[973, 184, 1784]`, while encoding
`" positive"` as one token `[17384]`.

`CandidatePaths` holds both encodings: `bare` is the answer exactly as written,
and `spaced` is the answer with one space in front. Encode each one together
with the prompt, the way the model would see the full text, and keep the tokens
after the prompt. The class stores token IDs you supply; it does not tokenize
text.

### A small example

For illustration, let `10` represent `"up"`, `11` represent `"war"`, `12`
represent `"d"`, and `20` represent `"down"`. The spaced spellings use single
IDs `30` (`" up"`), `31` (`" upward"`), and `40` (`" down"`). END IDs `99` and
`100` both mean stopping.

```pycon
>>> import math
>>> from credences.probabilities import TokenScores
>>> from credences.trie import CandidatePaths, build_trie
>>> forms = {
...     "up": CandidatePaths(bare=[10], spaced=[30]),
...     "upward": CandidatePaths(bare=[10, 11, 12], spaced=[31]),
...     "down": CandidatePaths(bare=[20], spaced=[40]),
... }
>>> forms["upward"].bare
(10, 11, 12)

```

The IDs are stored as tuples. Nothing checks that they spell the intended
words; that needs a tokenizer.

`build_trie` checks every spelling and plans the model calls, before any model
is run. The bare `"up"` path ends where the bare `"upward"` path continues, so the
model must be able to signal "stop" with END tokens.

```pycon
>>> plan = build_trie(forms, end_tokens=[100, 99, 99])
>>> plan.end_tokens
(99, 100)
>>> [(request.prefix, request.wanted) for request in plan.readouts]
[((), (10, 20, 30, 31, 40)), ((10,), (11, 99, 100))]

```

The plan contains **requests, not scores**:

| Prefix | Decision |
|---|---|
| `()` | Choose among all five first tokens; `10` starts both `"up"` and `"upward"` but is scored once |
| `(10,)` | Continue with `11`, or stop with END `99` or `100` |
| `(10, 11)` | Only `12` is possible, so no score is needed |

Repeated END IDs count once. The plan depends only on the token paths, so it
can be shown before measuring and reused for any prompt with the same paths.

Suppose your prompt encodes to the tokens `C`. Each request is made after
`C` **plus the full prefix**:

```pycon
>>> C = (700, 701)
>>> [(C + request.prefix, request.wanted) for request in plan.readouts]
[((700, 701), (10, 20, 30, 31, 40)), ((700, 701, 10), (11, 99, 100))]

```

The model answers each request with `TokenScores`: the requested **logits**
(raw scores, which can be positive) and its **log normalizer**, the logsumexp of
the logits over its whole vocabulary. Only the coverage diagnostic uses the
normalizer.

```pycon
>>> scores = {
...     (): TokenScores({10: 2.0, 20: 2.0, 30: 1.0, 31: 0.0, 40: 1.0}, log_normalizer=4.0),
...     (10,): TokenScores({11: 0.0, 99: 0.0, 100: 0.0}, log_normalizer=2.0),
... }
>>> raw = plan.score(scores, context_token_count=len(C))
>>> {label: round(math.exp(value), 6) for label, value in raw.credence_logprobs.items()}
{'up': 0.360332, 'upward': 0.163237, 'down': 0.476431}

```

Each answer's credence adds up its spellings:

```pycon
>>> [(path.spellings, round(math.exp(path.logprob), 6)) for path in raw.scored_paths["up"]]
[(('bare',), 0.2322), (('spaced',), 0.128132)]

```

| Label | Probability calculation |
|---|---|
| `"up"` | Token `10` (0.348) times stopping (either END token, 2/3), plus token `30` (0.128) |
| `"upward"` | Token `10` times continuing (1/3), with `12` then certain, plus token `31` (0.047) |
| `"down"` | Token `20` (0.348) plus token `40` (0.128) |

`score` adds log probabilities, which is the same as multiplying probabilities.
The results already sum to 1, with no rescaling.

`raw.coverage` is the model's unrestricted probability that its first answer
token is one of the five allowed ones. It is a diagnostic about the model's
habits, not about accuracy, and it never changes the credences:

```pycon
>>> round(raw.coverage, 6)
0.38856

```

## 2. From a raw readout to results and JSON

Continue with `raw` from the first section. It records every scored path, the
number of prompt tokens, and the coverage.

```pycon
>>> from credences.probabilities import measurement_from_raw
>>> result = measurement_from_raw(raw)
>>> result.top_labels
('down',)
>>> result.raw.scored_paths["upward"][0].token_ids
(10, 11, 12)
>>> round(result.top_label_confidence, 6), round(result.margin_confidence, 6)
(0.214647, 0.1161)

```

`measurement_from_raw` checks that the credences sum to 1, then computes the
summaries. It never rescales or rounds the credences.

`to_dict()` returns a copy you can save as JSON:

```pycon
>>> import json
>>> exported = result.to_dict()
>>> restored = json.loads(json.dumps(exported, allow_nan=False))
>>> restored["top_labels"]
['down']
>>> restored["raw"]["scored_paths"]["upward"][0]["token_ids"]
[10, 11, 12]

```

A measurement keeps its own copy of its inputs, so later changes to the
`RawReadout` do not affect it:

```pycon
>>> saved_down_log = result.raw.credence_logprobs["down"]
>>> raw.credence_logprobs["down"] = -math.inf
>>> result.raw.credence_logprobs["down"] == saved_down_log
True

```

## 3. Exact ties and caller-owned randomness

Here two labels tie for the highest credence. `top_label_weights` splits one
win equally between them, which is useful for counting winners across many
measurements. These weights are not credences.

```pycon
>>> from credences import choose_top_label, top_label_weights
>>> forms = {label: CandidatePaths((i,), (i + 10,)) for i, label in enumerate("ABC")}
>>> logits = {0: math.log(0.4), 1: math.log(0.4), 2: math.log(0.2)}
>>> logits |= dict.fromkeys((10, 11, 12), -math.inf)
>>> plan = build_trie(forms)
>>> tied = measurement_from_raw(
...     plan.score({(): TokenScores(logits, 0.0)}, context_token_count=7)
... )
>>> tied.top_labels
('A', 'B')
>>> top_label_weights(tied)
{'A': 0.5, 'B': 0.5}
>>> import random
>>> winner = choose_top_label(tied, rng=random.Random(7))
>>> winner in tied.top_labels
True

```

Measurements never break ties themselves. `choose_top_label` uses only the
generator you pass, so a seeded generator makes the choice reproducible.

## 4. Numerical rules worth remembering

Logits are raw scores and can be any real number; log probabilities are at
most 0. Each step's probabilities are a softmax over only the tokens allowed
there. A step with one possible token has probability 1 and needs no score,
like the step after `(10, 11)` in section 1.

When every allowed token scores `-inf`, the tokens share the probability
equally. That is equal per **token**, not per answer: answers that start with
the same token split that token's share. To keep this example short, each
answer has one path for both spellings.

```pycon
>>> forms = {
...     "upward": CandidatePaths((10, 11), (10, 11)),
...     "upstairs": CandidatePaths((10, 13), (10, 13)),
...     "down": CandidatePaths((20,), (20,)),
... }
>>> impossible = {
...     (): TokenScores({10: -math.inf, 20: -math.inf}, log_normalizer=0.0),
...     (10,): TokenScores({11: -math.inf, 13: -math.inf}, log_normalizer=0.0),
... }
>>> fallback = build_trie(forms).score(impossible, context_token_count=7)
>>> {label: round(math.exp(value), 6) for label, value in fallback.credence_logprobs.items()}
{'upward': 0.25, 'upstairs': 0.25, 'down': 0.5}

```

`score` checks every score before using any of them. A missing score, NaN, or
+inf raises `BackendReadoutError`.

A very small probability can round to 0.0, but its log credence stays finite,
so it remains distinct from an answer with exactly zero probability:

```pycon
>>> forms = {
...     "likely": CandidatePaths((1,), (2,)),
...     "tiny": CandidatePaths((3,), (4,)),
...     "impossible": CandidatePaths((5,), (6,)),
... }
>>> logits = {1: 0.0, 3: -1000.0} | dict.fromkeys((2, 4, 5, 6), -math.inf)
>>> tiny = measurement_from_raw(
...     build_trie(forms).score({(): TokenScores(logits, 0.0)}, context_token_count=7)
... )
>>> tiny.credences
{'likely': 1.0, 'tiny': 0.0, 'impossible': 0.0}
>>> tiny.to_dict()["raw"]["credence_logprobs"]
{'likely': 0.0, 'tiny': -1000.0, 'impossible': None}

```

In JSON, only exactly-zero log probabilities become `null`. Winners are found by
comparing exact log credences, so two answers whose probabilities happen to
round to the same float are not reported as tied.
