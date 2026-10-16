"""Model backends: the code that runs a model and reports its next-token scores."""

from .base import FullLogitsBackend

__all__ = ["FullLogitsBackend"]
