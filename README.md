# credences

Measure how a language model divides its belief among a fixed set of answers.

Ask a model a multiple-choice question, such as "Is this sentence positive,
negative, or neutral?" Rather than keeping only the answer it would write,
`credences` reads the model's next-token probabilities and returns a
probability for every allowed answer:

```python
{"positive": 0.91, "negative": 0.03, "neutral": 0.06}
```

This is the model's *credence vector*. It shows which answer the model prefers
and how strongly, and it is the input for confidence measures such as entropy
or the gap between the top two answers.

## Status

Early development. The package is not yet on PyPI.

- **Available now:** the scoring math and result types. Given the token IDs
  for each answer and a model's logits (raw next-token scores), `credences`
  computes each answer's probability, confidence summaries, and JSON output.
- **Not yet available:** tokenizing answers, loading a model, and a one-call
  measurement API. The planned API looks like this:

  ```python
  # Planned; not implemented yet.
  from credences import load_model, build_credoscope

  model = load_model("mlx-community/Qwen3-4B-4bit")  # local model via MLX
  scope = build_credoscope(model, candidates=["positive", "negative", "neutral"])
  result = scope.measure(prompt="Classify the sentiment of: 'Revenue beat expectations.' ...")
  result.credences
  ```

## How answers are scored

A model writes one token at a time, and an answer such as `"very positive"`
may be several tokens. To get one probability per answer:

1. **Encode each answer twice:** as written, and with a space in front.
   Tokenizers often split `"positive"` and `" positive"` into different tokens,
   and a model may expect either right after the prompt. For each answer,
   `credences` keeps whichever version's first token the model scores higher.
2. **Score only the allowed tokens.** At each step, the model's scores for the
   tokens the answers allow there are turned into probabilities among those
   tokens alone, ignoring the rest of the vocabulary.
3. **Multiply along each answer.** An answer's probability is the product of
   its steps. Answers that start with the same token share that step's
   probability instead of each counting it in full. When one answer is the
   start of another (`"up"` and `"upward"`), the model's choice to stop is read
   from its end-of-answer tokens.

The resulting probabilities sum to 1 over your answers. They describe how the
model divides its belief among the answers you gave, for that prompt; they are
not guaranteed to be calibrated.

## Try it

You need [uv](https://docs.astral.sh/uv/) and Python 3.14. From a clone of
this repository:

```bash
uv sync --group dev
```

The current version works on token IDs and logits you provide. Here, two
one-token answers use made-up token IDs and logits:

```pycon
>>> from credences import RawReadout
>>> from credences.probabilities import measurement_from_raw
>>> from credences.trie import build_trie
>>> plan = build_trie({"yes": (10,), "no": (20,)})  # "yes" is token 10, "no" is 20
>>> logs = plan.score({(): {10: 2.0, 20: 1.0}})  # the model's logits for each token
>>> result = measurement_from_raw(RawReadout(logs, dict(plan.paths), context_token_count=12))
>>> {label: round(p, 3) for label, p in result.credences.items()}
{'yes': 0.731, 'no': 0.269}
>>> result.top_labels
('yes',)

```

For multi-token answers, the with-and-without-space choice, shared first
tokens, and saving results, follow the
[walkthrough of the scoring core](docs/pure-core.md).

## What a measurement contains

`Measurement.credences` holds the probabilities. The other fields summarize
them, for K answers whose largest and second-largest probabilities are p1 and
p2:

| Field | Meaning |
|---|---|
| `top_labels` | Every answer tied for the highest probability; usually one |
| `entropy` | Shannon entropy in bits: 0 when certain, log2(K) when all answers are equal |
| `top_label_confidence` | How far the top answer is above a uniform guess: `(p1 - 1/K) / (1 - 1/K)` |
| `entropy_confidence` | How concentrated the whole distribution is: `1 - entropy / log2(K)` |
| `margin_confidence` | The top answer's lead over the runner-up: `p1 - p2` |

Each confidence summary runs from 0 to 1, and is 1 when one answer has all
the probability.
`measurement.raw` keeps the exact log probabilities, which stay meaningful even
for answers too unlikely to show as anything but 0.0, and the token IDs scored
for each answer. `measurement.to_dict()` gives a copy you can save as JSON.

Ties are never broken silently. Use `choose_top_label(result, rng=...)` to pick
one winner with your own random generator, or `top_label_weights(result)` to
split a win equally when counting results.

## Documentation

- [Walkthrough of the scoring core](docs/pure-core.md): runnable examples from
  token IDs to a saved result.
- API reference: the docstrings, for example `help(credences.trie)`. Their
  examples run as tests.
- [Conceptual note](confidence-notes.pdf): why the credence vector is the
  primary result and how the confidence summaries derive from it.

## Contributing

Run the checks before sending a change:

```bash
uv sync --locked --group dev
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run lint-imports
uv run pytest
uv build
```

`uv run pytest` needs no model weights. It also runs the examples in this
README, the walkthrough, and the docstrings.

These documents describe the design and how to work on it:

- [`docs/style.md`](docs/style.md): code and docstring conventions.
- [`tests/README.md`](tests/README.md): what each test file guards.
- [`SPEC.md`](SPEC.md): the full design of version 1, including the planned API.
- [`DECISIONS.md`](DECISIONS.md): why the design is shaped this way.
- [`PLAN.md`](PLAN.md): the build order.
- [`POSSIBLE_FUTURE_EXTENSIONS.md`](POSSIBLE_FUTURE_EXTENSIONS.md): ideas kept
  out of version 1 on purpose.
- [`AGENTS.md`](AGENTS.md): the working agreement for coding agents and
  maintainers, including how the main checkout is kept in sync.
- [`CHANGELOG.md`](CHANGELOG.md): changes so far.

## License

MIT. See [LICENSE](LICENSE).
