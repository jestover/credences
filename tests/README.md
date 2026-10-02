# Tests

`test_results.py` guards the result persistence contract: the exact JSON schema,
all tied winners, selected complete paths, detached exported containers, and
finite log credences preserved through probability underflow. It also prevents
malformed nonfinite values from escaping as invalid JSON.

Run with `uv run pytest`. No tokenizer, model weights, or backend are required.
Add the remaining contract-focused tests from `PLAN.md` and `SPEC.md` as each
implementation step is introduced.
