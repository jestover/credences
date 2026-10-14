# Small examples of the pure measurement core

These examples use **invented token IDs and logits**, not a real tokenizer or
model. They show the lower-level building blocks that the planned model-backed
API will run for you; that API is not implemented yet. Run the snippets on this
page in order.

`>>>` and `...` mark interactive Python input; unprefixed lines are expected
output. To turn a snippet into a script, remove the prompts and output lines.

Outputs are rounded here only for readability; the library itself never rounds.
`uv run pytest` runs every example on this page, so they stay in step with the code.

## 1. From two forms per candidate to a scored trie

### Why keep two token paths?

Leading spaces can change how a tokenizer represents an answer. A hypothetical
tokenizer might split `"positive"` into `"pos"`, `"it"`, and `"ive"` with IDs
`[973, 184, 1784]`, while encoding `" positive"` as one token `[17384]`. The two
forms can therefore differ in both length and token IDs, not just their first
piece.

`CandidatePaths` holds both encodings: `bare` is the answer exactly as
written, and `spaced` is the answer with one space in front. Encode both as
they would appear right after the prompt. For each answer, the model's
first-token logits decide which version is scored. That choice only decides how
the answer is represented; it does not change the answer's probability. The
class stores token IDs you supply; it does not tokenize text.

### A small example

For illustration, let `10` represent `"up"`, `11` represent `"war"`, `12`
represent `"d"`, and `20` represent `"down"`. The spaced forms use single IDs
`30`, `31`, and `40`. END IDs `99` and `100` both mean stopping.

```pycon
>>> import math
>>> from credences.trie import (
...     CandidatePaths, build_trie, first_token_ids, select_paths,
...     validate_potential_paths,
... )
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

Before asking the model for anything, check **both** encodings of every answer.
The bare `"up"` path ends where the bare `"upward"` path continues, so the
model must be able to signal "stop" with END tokens, here IDs `99` and `100`.

```pycon
>>> end_tokens = [100, 99, 99]
>>> validate_potential_paths(forms, end_tokens=end_tokens)
>>> first_token_ids(forms)
(10, 20, 30, 31, 40)

```

Six encodings need only five first-token scores, because two start with `10`.
Suppose your prompt encodes to the tokens `C`. Ask the model for its logits
right after `C`, with no answer tokens added.

```pycon
>>> C = (700, 701)
>>> first_scores = {10: 2.0, 20: 2.0, 30: 1.0, 31: 0.0, 40: 1.0}
>>> selected = select_paths(forms, first_scores)
>>> selected
{'up': (10,), 'upward': (10, 11, 12), 'down': (20,)}

```

These are **logits** (raw scores), so they can be positive. Here the bare
version wins for every answer. Each answer keeps its whole chosen encoding, and
on an exact tie the bare version wins.

Plan the scoring for the chosen paths:

```pycon
>>> plan = build_trie(selected, end_tokens=end_tokens)
>>> plan.paths
(('up', (10,)), ('upward', (10, 11, 12)), ('down', (20,)))
>>> plan.end_tokens
(99, 100)
>>> [(request.prefix, request.wanted) for request in plan.readouts]
[((), (10, 20)), ((10,), (11, 99, 100))]

```

The plan contains **requests, not scores**:

| Prefix | Decision |
|---|---|
| `()` | Choose `10` or `20`; `10` starts both `"up"` and `"upward"` but is scored once |
| `(10,)` | Continue with `11`, or stop with END `99` or `100` |
| `(10, 11)` | Only `12` is possible, so no score is needed |

An answer that no other answer continues needs no stopping score. Repeated END
IDs count once.

The scores right after the prompt are already in hand, so only the request at
`(10,)` needs another model call. Make it after the prompt **plus the full
prefix**:

```pycon
>>> [(C + request.prefix, request.wanted) for request in plan.additional_readouts]
[((700, 701, 10), (11, 99, 100))]
>>> readouts = {
...     (): first_scores,
...     (10,): {11: 0.0, 99: 0.0, 100: 0.0},
... }
>>> logs = plan.score(readouts)
>>> {label: round(math.exp(value), 6) for label, value in logs.items()}
{'up': 0.333333, 'upward': 0.166667, 'down': 0.5}

```

`plan.readouts` says where scores are needed. The `readouts` argument to
`score` supplies them, keyed by the prefix after the prompt (not including `C`).

Each label's probability is the product of the steps along its path:

| Label | Probability calculation |
|---|---|
| `"up"` | First token `1/2` times stopping probability (either END token) `2/3` |
| `"upward"` | First token `1/2` times continuing probability `1/3`; the last step, `12`, has probability 1 |
| `"down"` | First token `1/2`; nothing else starts with `20` |

`score` adds log probabilities, which is the same as multiplying
probabilities. The results already sum to 1, with no rescaling. A plan stores
no prompt or scores, so you can reuse it for another prompt whenever
`select_paths` chooses the same paths.

## 2. From log credences to results and JSON

Continue with `logs`, `plan`, and `C` from the first section. The raw result
records the token path scored for each label and the number of prompt tokens.

```pycon
>>> from credences import RawReadout
>>> from credences.probabilities import measurement_from_raw
>>> raw = RawReadout(logs, dict(plan.paths), context_token_count=len(C))
>>> result = measurement_from_raw(raw)
>>> result.top_labels
('down',)
>>> result.raw.scored_token_ids["upward"]
(10, 11, 12)
>>> round(result.top_label_confidence, 6), round(result.margin_confidence, 6)
(0.25, 0.166667)

```

`measurement_from_raw` checks that the credences sum to 1 and then computes
the summaries. It never rescales or rounds the credences.

`to_dict()` returns a copy you can save as JSON:

```pycon
>>> import json
>>> exported = result.to_dict()
>>> restored = json.loads(json.dumps(exported, allow_nan=False))
>>> restored["top_labels"]
['down']
>>> restored["raw"]["scored_token_ids"]["upward"]
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

`RawReadout` itself does not copy, so `raw.credence_logprobs` and `logs` are
the same dictionary; that edit changed `logs` too.

## 3. Exact ties and caller-owned randomness

Here two labels tie for the highest credence. `top_label_weights` splits one
win equally between them, which is useful for counting winners across many
measurements. These weights are not credences.

```pycon
>>> from credences import choose_top_label, top_label_weights
>>> tied = measurement_from_raw(RawReadout(
...     {"A": math.log(0.4), "B": math.log(0.4), "C": math.log(0.2)},
...     {"A": (10,), "B": (20,), "C": (30,)}, 7,
... ))
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
most 0. `masked_logprobs`, which scores a single step, returns log
probabilities; exponentiate them to get ordinary probabilities.

```pycon
>>> from credences.probabilities import masked_logprobs
>>> branch = masked_logprobs([10, 20], {10: 2.0, 20: 1.0})
>>> {token: round(math.exp(value), 6) for token, value in branch.items()}
{10: 0.731059, 20: 0.268941}
>>> masked_logprobs([12], {})  # One allowed token is forced, without a score.
{12: 0.0}
>>> fallback = masked_logprobs([10, 20], {10: -math.inf, 20: -math.inf})
>>> [round(math.exp(value), 6) for value in fallback.values()]
[0.5, 0.5]

```

When every allowed token scores `-inf`, the tokens share the probability
equally. That is equal per **token**, not per answer: answers that start with
the same token split that token's share. A missing score, NaN, or +inf raises
`BackendReadoutError`.

A very small probability can round to 0.0, but its log credence stays finite,
so it remains distinct from an answer with exactly zero probability:

```pycon
>>> tiny = measurement_from_raw(RawReadout(
...     {"likely": 0.0, "tiny": -1000.0, "impossible": -math.inf},
...     {"likely": (10,), "tiny": (20,), "impossible": (30,)}, 7,
... ))
>>> tiny.credences
{'likely': 1.0, 'tiny': 0.0, 'impossible': 0.0}
>>> tiny.to_dict()["raw"]["credence_logprobs"]
{'likely': 0.0, 'tiny': -1000.0, 'impossible': None}

```

In JSON, only exactly-zero log credences become `null`. Winners are found by
comparing exact log credences, so two answers whose probabilities happen to
round to the same float are not reported as tied.
