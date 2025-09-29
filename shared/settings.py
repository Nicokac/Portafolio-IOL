"""Cache and application settings exposed for cross-module use.

This module centralizes access to the configuration values used across
services and infrastructure layers. Values are sourced from environment
variables, `streamlit` secrets or ``config.json`` via ``shared.config``.
"""
from __future__ import annotations

from shared.config import settings as _config_settings

# Re-export the shared Settings instance so existing imports keep working.
settings = _config_settings

# Cache-related configuration. Importers can rely on these names instead of
# scattering magic numbers or environment lookups throughout the codebase.
cache_ttl_portfolio: int = settings.cache_ttl_portfolio
cache_ttl_last_price: int = settings.cache_ttl_last_price
cache_ttl_fx: int = settings.cache_ttl_fx
cache_ttl_quotes: int = settings.cache_ttl_quotes
cache_ttl_yf_indicators: int = settings.cache_ttl_yf_indicators
cache_ttl_yf_history: int = settings.cache_ttl_yf_history
cache_ttl_yf_fundamentals: int = settings.cache_ttl_yf_fundamentals
cache_ttl_yf_portfolio_fundamentals: int = settings.cache_ttl_yf_portfolio_fundamentals
quotes_hist_maxlen: int = settings.quotes_hist_maxlen
max_quote_workers: int = settings.max_quote_workers
yahoo_fundamentals_ttl: int = settings.YAHOO_FUNDAMENTALS_TTL
yahoo_quotes_ttl: int = settings.YAHOO_QUOTES_TTL
min_score_threshold: int = settings.min_score_threshold
max_results: int = settings.max_results

# Backwards compatibility for legacy imports
YAHOO_FUNDAMENTALS_TTL: int = yahoo_fundamentals_ttl
YAHOO_QUOTES_TTL: int = yahoo_quotes_ttl
MIN_SCORE_THRESHOLD: int = min_score_threshold
MAX_RESULTS: int = max_results

# Feature flags
FEATURE_OPPORTUNITIES_TAB: bool = bool(
    getattr(settings, "FEATURE_OPPORTUNITIES_TAB", False)
)

__all__ = [
    "settings",
    "cache_ttl_portfolio",
    "cache_ttl_last_price",
    "cache_ttl_fx",
    "cache_ttl_quotes",
    "cache_ttl_yf_indicators",
    "cache_ttl_yf_history",
    "cache_ttl_yf_fundamentals",
    "cache_ttl_yf_portfolio_fundamentals",
    "quotes_hist_maxlen",
    "max_quote_workers",
    "yahoo_fundamentals_ttl",
    "yahoo_quotes_ttl",
    "min_score_threshold",
    "max_results",
    "YAHOO_FUNDAMENTALS_TTL",
    "YAHOO_QUOTES_TTL",
    "MIN_SCORE_THRESHOLD",
    "MAX_RESULTS",
    "FEATURE_OPPORTUNITIES_TAB",
]
