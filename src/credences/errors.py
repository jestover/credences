"""Exceptions raised by credences."""


class CredencesError(Exception):
    """Base class for errors specific to credences.

    Catch this to handle every credences-specific failure in one place.
    Invalid arguments, such as an empty list of candidates, raise the
    standard ValueError or TypeError instead.
    """


class BackendReadoutError(CredencesError):
    """The model's scores are incomplete or invalid.

    Raised when a token score you asked for is missing, is not a number, or
    is NaN or +inf. A missing score is never treated as zero probability. If
    a model backend raises this, check that it returns a score for every
    requested token rather than only its highest-scoring tokens. A score of
    ``-inf`` is valid and means the token is impossible.

    Examples:
        The plan asks for tokens 10, 11, 20, and 21, but the scores leave out 21:

        >>> from credences.probabilities import TokenScores
        >>> from credences.trie import CandidatePaths, build_trie
        >>> plan = build_trie(
        ...     {"yes": CandidatePaths((10,), (11,)), "no": CandidatePaths((20,), (21,))}
        ... )
        >>> scores = {(): TokenScores({10: 2.0, 11: 1.0, 20: 0.5}, log_normalizer=3.0)}
        >>> plan.score(scores, context_token_count=7)
        Traceback (most recent call last):
            ...
        credences.errors.BackendReadoutError: Missing requested score for token 21
    """
