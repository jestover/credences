"""The interface every model backend implements."""

from collections.abc import Sequence
from typing import Protocol

from ..probabilities import TokenScores
from ..tokenization import Tokenizer


class FullLogitsBackend(Protocol):
    """A model that can score any next tokens after any sequence of tokens.

    credences asks a backend one kind of question: given these token IDs (a
    prompt, possibly followed by part of an answer), what logits does the
    model give these particular next tokens? Answering it requires direct
    access to the model's output scores, which is why local models are
    supported and hosted APIs that return only the top few tokens are not.

    A backend reports the model's raw logits, nothing else. It must return a
    logit for every requested token, including very unlikely ones, and must
    not sample, keep only the highest-scoring tokens, or adjust scores with a
    temperature or penalties. Use ``-inf`` for a token the model can never
    produce. credences checks every score it receives and raises
    BackendReadoutError for a missing, NaN, or +inf score instead of
    guessing.
    """

    def tokenizer(self) -> Tokenizer:
        """Return the tokenizer that matches this backend's model.

        Returns:
            The tokenizer whose token IDs this backend accepts and scores.
        """
        ...

    def next_token_logits(self, token_ids: Sequence[int], wanted: Sequence[int]) -> TokenScores:
        """Return the model's logits for the requested next tokens.

        Args:
            token_ids: The complete input so far: prompt tokens followed by
                any answer tokens.
            wanted: The token IDs to score at the next position.

        Returns:
            A logit for every ID in ``wanted``, as Python floats (extra IDs are
            allowed and ignored), and the logsumexp of the logits over the
            whole vocabulary, used only for the coverage diagnostic.
        """
        ...
