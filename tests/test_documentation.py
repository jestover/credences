"""Runnable examples must stay accurate without importing optional runtimes."""

import doctest
from pathlib import Path

import pytest

from credences import errors, probabilities, trie


@pytest.mark.parametrize(
    "module", [errors, probabilities, trie], ids=lambda module: module.__name__
)
def test_core_docstring_examples_execute_without_drift(module):
    result = doctest.testmod(module)

    assert result.attempted > 0, "No examples were discovered"
    assert result.failed == 0


@pytest.mark.parametrize("relative_path", ["README.md", "docs/pure-core.md"])
def test_markdown_examples_execute_without_drift(relative_path):
    path = Path(__file__).resolve().parents[1] / relative_path
    result = doctest.testfile(str(path), module_relative=False)

    assert result.attempted > 0, "No Markdown examples were discovered"
    assert result.failed == 0
