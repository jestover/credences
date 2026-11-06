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

- **Available now:** the scoring math and the result types described below.
  They work on model scores you supply; the
  [walkthrough](docs/pure-core.md) shows them step by step. The interfaces a
  model backend will implement are defined too.
- **Not yet available:** loading a model and the one-call measurement API. The
  planned API looks like this:

  ```python
  # Planned; not implemented yet.
  from credences import load_model, build_credoscope

  model = load_model("mlx-community/Qwen3-4B-4bit")  # local model via MLX
  scope = build_credoscope(model, candidates=["positive", "negative", "neutral"])
  result = scope.measure(prompt="Classify the sentiment of: 'Revenue beat expectations.' ...")
  result.credences
  ```

## What the numbers mean

The credences sum to 1 across your answers. They describe how the model divides
its probability among the answers you listed, for that prompt: in effect, how
often it would give each one if it had to answer with one of them. They are not
guaranteed to be calibrated, and a different wording of the prompt can give
different credences.

Only your answers, as you wrote them, are counted. To a model, `"Yes"` and
`"yes"` are different answers, so list your answers in the form your prompt
asks for. [How answers become tokens](docs/tokenization.md) explains how
answers are matched to the model's output, and why.

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
for answers too unlikely to show as anything but 0.0, and a breakdown of how
each answer's probability was put together. It also reports **coverage**: how
likely the model was, before being limited to your answers, to start one of
them. Coverage describes the model's habits, not the measurement's accuracy. A
model that likes to open with "The" can have low coverage and exactly the same
credences as one that answers directly. `measurement.to_dict()` gives a copy
you can save as JSON.

Ties are never broken silently. Use `choose_top_label(result, rng=...)` to pick
one winner with your own random generator, or `top_label_weights(result)` to
split a win equally when counting results.

## Documentation

- [How answers become tokens](docs/tokenization.md): what exactly is scored,
  written for readers new to how language models work.
- [Walkthrough of the scoring core](docs/pure-core.md): runnable examples from
  token IDs to a saved result.
- API reference: the docstrings, for example `help(credences.trie)`. Their
  examples run as tests.
- [Conceptual note](confidence-notes.pdf): why the credence vector is the
  primary result and how the confidence summaries derive from it.

## Contributing

You need [uv](https://docs.astral.sh/uv/) and Python 3.14. Run the checks
before sending a change:

```bash
uv sync --locked --group dev
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run lint-imports
uv run pytest
uv build
```

`uv run pytest` needs no model weights. It also runs the examples in the
walkthrough and the docstrings.

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
