"""Clients for macroeconomic and sector-level data providers."""

from .fred_client import FredClient, FredSeriesObservation, MacroAPIError, MacroAuthenticationError, MacroRateLimitError

__all__ = [
    "FredClient",
    "FredSeriesObservation",
    "MacroAPIError",
    "MacroAuthenticationError",
    "MacroRateLimitError",
]
