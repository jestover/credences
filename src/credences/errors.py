"""Errors shared by the pure core and eventual model integrations."""


class CredencesError(Exception):
    """Base class for credences-specific failures."""


class BackendReadoutError(CredencesError):
    """A requested token score is missing or numerically invalid.

    Missing scores are errors, not zero weights:

    >>> from credences.probabilities import validate_readout
    >>> validate_readout([10, 20], {10: 2.0})
    Traceback (most recent call last):
        ...
    credences.errors.BackendReadoutError: Missing requested score for token 20
    """
