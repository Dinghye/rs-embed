class RSEmbedError(Exception):
    """Base exception for all rs-embed errors."""

    pass


class SpecError(RSEmbedError):
    """Raised when an input spec is invalid or inconsistent."""

    pass


class ProviderError(RSEmbedError):
    """Raised when a data provider fails or is unsupported."""

    pass


class ModelError(RSEmbedError):
    """Raised for model loading, inference, or capability errors."""

    pass
