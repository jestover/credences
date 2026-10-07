# Tests

`test_results.py` guards the result persistence contract: the exact JSON schema,
all tied winners, selected complete paths, detached exported containers, and
finite log credences preserved through probability underflow. It also prevents
malformed nonfinite values from escaping as invalid JSON.

`test_probabilities.py` compares selected-path chain-rule calculations with an
independent Decimal reference. The production selector and trie consume scores
for generated paths and golden examples; the oracle independently scans paths
and multiplies probabilities. Coverage includes shared prefixes, changing
selected branches, grouped END events, forcing, token-uniform fallback, missing
scores, zero incoming mass, and extreme finite float32 logits.

`test_trie.py` checks path ownership, potential versus selected stopping
boundaries, exact form ties, immutable plan ownership, and stateless reuse.
A thin declared-logits fake records full contexts and wanted token ids to guard
root reuse, distinct-id requests, forced-prefix preservation, and omitted leaf
or singleton queries. Long forced chains are not recursion-limited.

`test_summaries.py` checks confidence definitions and boundary cases, exact-log
winners versus probability-rounding ties, normalization without label-level
renormalization, candidate-order invariance, and caller-owned randomness.

`test_documentation.py` executes doctest examples from explicitly listed
backend-free modules and the Markdown walkthrough under `docs/`. These guard
against documentation drift without enabling project-wide source imports that
could accidentally load future optional model runtimes during collection.

Run with `uv run pytest`. No tokenizer, model weights, or backend are required.
Extend these behavioral contracts as tokenization and backend implementations
are introduced.
