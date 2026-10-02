# Tests

`test_results.py` guards the result persistence contract: the exact JSON schema,
all tied winners, selected complete paths, detached exported containers, and
finite log credences preserved through probability underflow. It also prevents
malformed nonfinite values from escaping as invalid JSON.

`test_probabilities.py` compares selected-path chain-rule calculations with an
independent Decimal reference before the production trie exists. Its test-only
adapter drives the pure primitives; replace it with the actual selector and
trie when those are implemented. Coverage includes shared prefixes, changing
selected branches, grouped END events, forcing, token-uniform fallback, missing
scores, zero incoming mass, and extreme finite float32 logits.

`test_summaries.py` checks confidence definitions and boundary cases, exact-log
winners versus probability-rounding ties, normalization without label-level
renormalization, candidate-order invariance, and caller-owned randomness.

Run with `uv run pytest`. No tokenizer, model weights, or backend are required.
Extend these behavioral contracts as the selector, production trie,
tokenization, and backend implementations are introduced.
