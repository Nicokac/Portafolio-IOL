"""Shared exception types for cross-layer error handling."""

class NetworkError(Exception):
    """Raised when connectivity issues prevent reaching a remote service."""


class ExternalAPIError(NetworkError):
    """Raised when an external API call fails despite having network access."""
