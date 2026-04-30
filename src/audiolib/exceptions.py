"""
audiolib.exceptions — Error types for the audiolib API.
"""


class AudiolibError(Exception):
    """Base exception for audiolib."""


class ParameterError(AudiolibError):
    """Raised when an invalid parameter is passed to an audiolib function."""
