"""Common exception hierarchy used across the application."""


class PortfolioIOLError(Exception):
    """Base class for custom errors raised by the application."""


class AuthenticationError(PortfolioIOLError):
    """Base class for authentication related errors."""


class InvalidCredentialsError(AuthenticationError):
    """Raised when the provided credentials are rejected by the API."""


class NetworkError(PortfolioIOLError):
    """Raised when a request fails due to connectivity problems."""


class TimeoutError(NetworkError):
    """Raised when a request to an external service exceeds the timeout."""


__all__ = [
    "PortfolioIOLError",
    "AuthenticationError",
    "InvalidCredentialsError",
    "NetworkError",
    "TimeoutError",
]
