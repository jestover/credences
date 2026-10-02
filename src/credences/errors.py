"""Errors shared by the pure core and eventual model integrations."""


class CredencesError(Exception):
    """Base class for credences-specific failures."""


class BackendReadoutError(CredencesError):
    """A requested token score is missing or numerically invalid."""
