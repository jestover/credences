# Code and documentation style

The formatter and checks in `pyproject.toml` are the mechanical source of truth.
Use a small, explicit rule set rather than enabling every available rule.
This project uses Google-style **docstrings**, not the entire Google Python
coding guide.

## Formatting and names

- Run `uv run ruff format .`; do not maintain a competing manual layout.
- Use four-space indentation, double-quoted strings, and a 100-column target.
  The formatter also formats Python examples inside docstrings. Long literals
  and expected example output are not split just to satisfy a hard line limit.
- Use `snake_case` for functions and variables, `PascalCase` for classes and type
  aliases, and `UPPER_SNAKE_CASE` for constants.
- Let Ruff sort imports. Keep standard-library, third-party, and package imports
  separate. Existing same-package relative imports are fine.
- Use modern Python syntax appropriate for Python 3.14. Do not rewrite clear
  numerical code merely to shorten it.

## Google-style docstrings

Start with what the object or operation is for. Explain domain terms and why the
caller needs it before listing the interface. Function and method summaries use
imperative wording such as “Choose” or “Return” and end in a period. Class and
property summaries describe the object or value.

Use these sections when applicable:

- `Args:` describes arguments by their exact names, excluding `self` and `cls`.
- `Returns:` describes the result's meaning; use `Yields:` for generators.
  Omit it for constructors and other functions returning only `None`.
- `Raises:` documents intentional validation failures and other exceptions that
  callers should handle, not every incidental internal exception.
- `Attributes:` explains public record fields and their meaning.
- `Examples:` gives small, representative usage examples. Mark invented token
  IDs as illustrative; do not imply they belong to a real tokenizer.

Types belong in annotations. Do not repeat them in prose unless the description
adds constraints, units, shape, or other information the annotation cannot say.
In particular, distinguish raw logits, probabilities, natural-log probabilities,
and entropy in bits.

A representative function docstring has this structure:

```python
"""Choose one complete answer representation per candidate.

Compare the two forms' first-token logits after the same fixed context.
Exact ties prefer the bare form.

Args:
    forms: The bare and additional-space paths for each candidate.
    first_scores: Raw logits for every distinct first token in those paths.

Returns:
    Each candidate's selected complete token path, preserving candidate order.

Raises:
    BackendReadoutError: A requested first-token score is missing or invalid.
    ValueError: Candidate labels or path ownership are invalid.
"""
```

Do not add empty sections or repeat every field description in the constructor.
A short constructor docstring may refer to meanings already explained by its
class. Private helpers can have short descriptions when their contract is simple.

Non-obvious implementation rationale belongs in sparse comments near the code.
Do not turn docstrings or comments into a line-by-line implementation tutorial.
Do not refer users to temporary implementation plans.

Runnable usage examples remain checked by `uv run pytest`. Round displayed
output only, never values before scoring or comparing exact winners.

## What the tools check

### Ruff

The selected rules cover:

| Rules | Purpose |
|---|---|
| `E4`, `E7`, `E9`, `F` | Structural errors, undefined names, unused imports, and related mistakes |
| `I`, `UP`, `N` | Import ordering, modern Python, and naming conventions |
| `B` | Common bugs such as mutable default arguments |
| `ANN` | Explicit function and method annotations in shipped library code |
| `D`, `D400`, `D401` | Google docstring structure, summary punctuation, and function summary mood |
| `RUF100` | Suppressions that are no longer needed |

Tests retain the ordinary lint checks but are exempt from `ANN` and `D`: fixtures
and helpers do not need repetitive documentation, and negative tests deliberately
construct malformed values.

Ruff can enforce structure, not whether documentation is helpful or truthful.
Review still needs to check purpose, terminology, examples, and caller contracts.

### ty

Run `uv run ty check`. The normal target is `src/`, including public interfaces
and internal library helpers, under Python 3.14 semantics.

Keep annotations accurate and useful. Do not replace real types with `Any` or
blanket ignores just to silence a diagnostic. A necessary suppression must name
the diagnostic and explain why it is justified; unused ty ignores are errors.
Warnings fail CI too.

Tests are outside the default type-checking target because many intentionally
violate input types to exercise runtime validation. This is not a runtime
validation exemption: tests still run, and finite-score, normalized-mass, and
token-boundary checks remain essential.

## Canonical checks

```bash
uv sync --locked --group dev
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run lint-imports
uv run pytest
uv build
```

The lockfile pins tool versions used in CI. Runtime integrations stay optional;
style and type-checking tools are development dependencies only.

## References

- [Google Python Style Guide: docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [PEP 257: docstring conventions](https://peps.python.org/pep-0257/)
- [Ruff configuration](https://docs.astral.sh/ruff/configuration/)
- [ty configuration](https://docs.astral.sh/ty/configuration/)
