"""Shared application error hierarchy."""

from __future__ import annotations


class AppError(Exception):
    """Base exception for application-specific errors."""


class NetworkError(AppError):
    """Represents network-related failures."""


class InvalidCredentialsError(AppError):
    """Raised when provided credentials are invalid."""


class TimeoutError(NetworkError):
    """Raised when a network operation times out."""


class ExternalAPIError(AppError):
    """Raised when an external API returns an error."""


__all__ = [
    "AppError",
    "NetworkError",
    "InvalidCredentialsError",
    "TimeoutError",
    "ExternalAPIError",
]
