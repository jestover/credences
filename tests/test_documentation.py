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


def test_pure_core_walkthrough_executes_without_drift():
    path = Path(__file__).resolve().parents[1] / "docs" / "pure-core.md"
    result = doctest.testfile(str(path), module_relative=False)

    assert result.attempted > 0, "No walkthrough examples were discovered"
    assert result.failed == 0
