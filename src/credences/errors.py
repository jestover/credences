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
        >>> from credences.probabilities import validate_readout
        >>> validate_readout([10, 20], {10: 2.0})
        Traceback (most recent call last):
            ...
        credences.errors.BackendReadoutError: Missing requested score for token 20
    """
