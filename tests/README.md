# Tests

Run them with `uv run pytest`. No model weights or downloads are needed. Each
test is named for the mistake it would catch.

| File | What it protects |
|---|---|
| `test_probabilities.py` | The scoring math. A property test generates many answer sets and scores, and checks the library's results against a separate, straightforward high-precision calculation. Other tests cover very large and very small scores, END tokens, and missing or invalid scores. |
| `test_trie.py` | Planning which scores to request: shared first tokens are scored once, steps with only one possible token are not requested, every request uses the full input so far, and bad answer sets (identical paths, missing END tokens) are rejected before the model is called. |
| `test_summaries.py` | The confidence summaries, exact tie handling, the check that credences sum to 1, and the tie-breaking helpers. |
| `test_results.py` | Saving results as JSON: the field layout, all tied winners, and log credences for very unlikely answers. |
| `test_types.py` | Chat validation: a prompt string or a lone message is not mistaken for a chat, malformed messages are rejected, and assistant prefills are kept exactly, including trailing whitespace. |
| `test_backend_contract.py` | Scoring through the backend interface: a backend that returns only its highest-scoring tokens is rejected instead of producing renormalized, wrong credences. |
| `test_documentation.py` | Every example in the README, `docs/pure-core.md`, and the docstrings still runs and prints what it shows. |

`fakes.py` holds a small tokenizer and backend with declared behavior, used to
drive the library through the same interfaces a real model will implement.
`uv run ty check` includes it, so the fakes cannot drift from those interfaces.
