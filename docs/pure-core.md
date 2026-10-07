# Small examples of the pure measurement core

These examples use **invented token IDs and logits**, not a real tokenizer or
model. They show the internal building blocks; the model-backed measurement API
is not implemented yet. Run the snippets on this page in order.

`>>>` and `...` mark interactive Python input; unprefixed lines are expected
output. To turn a snippet into a script, remove the prompts and output lines.

Outputs are rounded only for readability. The library preserves unrounded
values and natural-log credences. `uv run pytest` executes this walkthrough and
the examples in the core docstrings, so they cannot quietly drift from the code.

## 1. From two forms per candidate to a scored trie

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

The constructor validates IDs and stores immutable tuples. It does not verify
that those IDs decode to the stated strings; that requires a tokenizer.

Before asking for scores, preflight **both** forms. The bare `"up"` path ends
where the bare `"upward"` path continues, so usable END IDs are required.

```pycon
>>> end_tokens = [100, 99, 99]
>>> validate_potential_paths(forms, end_tokens=end_tokens)
>>> first_token_ids(forms)
(10, 20, 30, 31, 40)

```

Six forms need only five distinct first-token scores: two share `10`.
Suppose the caller's fixed token context is `C`. Initial logits are requested
at `C`, without any candidate appended.

```pycon
>>> C = (700, 701)
>>> first_scores = {10: 2.0, 20: 2.0, 30: 1.0, 31: 0.0, 40: 1.0}
>>> selected = select_paths(forms, first_scores)
>>> selected
{'up': (10,), 'upward': (10, 11, 12), 'down': (20,)}

```

These are **logits**, which may be positive. Bare wins each comparison here.
Selection keeps the complete suffix and adds no probability factor. Exact
form-score ties also choose bare.

Compile only the selected paths:

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
| `()` | Choose `10` or `20`; `10` is counted once despite two labels sharing it |
| `(10,)` | Continue with `11`, or stop with END `99` or `100` |
| `(10, 11)` | Force `12`; no score needed |

Leaves stop without another query. END IDs are deduplicated before normalization.

The root reuses the initial scores, so only the non-root branch needs another
readout. Request it at the **unchanged context plus the full prefix**:

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

`plan.readouts` describes where comparisons are needed. The `readouts` argument
supplies the actual logits at those positions; its keys are prefixes relative
to `C`, not full contexts.

Inside scoring, each branch becomes a dictionary of token log probabilities.
The method then collects only the contributions belonging to each label:

| Label | Probability calculation |
|---|---|
| `"up"` | Root weight `1/2` times grouped stopping probability `2/3` |
| `"upward"` | Root weight `1/2` times continuation probability `1/3`; final `12` is forced |
| `"down"` | Root weight `1/2`; the path is a leaf |

The method adds logs rather than multiplying probabilities. Final probabilities
already sum to 1; there is no label-level renormalization. A plan contains no
prompt or score state and can score new readouts independently.

## 2. From log credences to results and JSON

Continue with `logs`, `plan`, and `C` from the first section. The raw result records
selected complete paths and the length of the fixed context—not the request
plan, discarded alternatives, or a generated inference trace.

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

The result builder checks that the log credences imply normalized mass. It
derives the vector and summaries without rounding or renormalizing the vector.

`to_dict()` makes a detached, JSON-safe snapshot:

```pycon
>>> import json
>>> exported = result.to_dict()
>>> restored = json.loads(json.dumps(exported, allow_nan=False))
>>> restored["top_labels"]
['down']
>>> restored["raw"]["scored_token_ids"]["upward"]
[10, 11, 12]

```

The result dataclasses prevent field reassignment, but their dictionaries remain
mutable. The result builder copies input mappings so edits to the original raw
object do not invalidate already-computed summaries:

```pycon
>>> saved_down_log = result.raw.credence_logprobs["down"]
>>> raw.credence_logprobs["down"] = -math.inf
>>> result.raw.credence_logprobs["down"] == saved_down_log
True

```

The original raw object shares the supplied `logs` dictionary, so that original
dictionary changes too. The measurement owns a separate snapshot.

## 3. Exact ties and caller-owned randomness

Here both winning labels have identical log credences. Counting weights split
one winner-count equally; they are not the original candidate probabilities.

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

The package does not select a winner during measurement. The optional helper
uses the supplied RNG, not global randomness, and does not alter the result.

## 4. Numerical rules worth remembering

Logits are unnormalized scores. Normalized log probabilities are <= 0.
The branch helper returns logs; exponentiate only when ordinary probabilities
are needed.

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

All-`-inf` fallback is uniform over **distinct tokens**, not final labels.
Missing requested scores, NaN, and positive infinity remain errors.

Very small probabilities can underflow, but their finite log credences remain
distinct from exact zero:

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

Finite logs stay finite in JSON; only exact-zero logs become `null`. Winning
labels are determined from exact computed log values, not rounded probabilities.
