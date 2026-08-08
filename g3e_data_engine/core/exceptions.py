"""
Custom exceptions for g3e-data-engine.

Kept in their own module (rather than inline in config.py or credentials.py)
so both can import from here without a circular import.
"""


class G3EDataEngineError(Exception):
    """Base class for all engine-specific errors."""


class SourceConfigError(G3EDataEngineError):
    """
    Raised when an enabled dataset source isn't safe/ready to download from —
    missing repo/project reference, or an unverified license. Raised by
    EngineConfig.validate_sources_ready(), always before any network call.
    """


class MissingCredentialError(G3EDataEngineError):
    """
    Raised when a source's downloader (or the HF uploader) needs a token/API
    key that isn't set in the environment or a .env file. See
    g3e_data_engine/core/credentials.py.
    """
