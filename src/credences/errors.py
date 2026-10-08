"""Exceptions applications can handle when credence computation fails."""


class CredencesError(Exception):
    """Base exception for handling package-specific measurement failures.

    Catch this base class when you do not need a separate handler for each
    credences exception type. Ordinary validation errors such as ValueError
    are not part of this hierarchy.
    """


class BackendReadoutError(CredencesError):
    """Required token scores are missing or cannot be used as logits.

    Check that the supplied readout contains every requested token ID and real
    numeric scores. If using a backend, it must not truncate the result to its
    top-scoring tokens. NaN and positive infinity are invalid; negative infinity
    is a valid zero weight.

    Missing scores are errors, not zero weights:

    >>> from credences.probabilities import validate_readout
    >>> validate_readout([10, 20], {10: 2.0})
    Traceback (most recent call last):
        ...
    credences.errors.BackendReadoutError: Missing requested score for token 20
    """
