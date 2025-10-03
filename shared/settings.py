"""Cache and application settings exposed for cross-module use.

This module centralizes access to the configuration values used across
services and infrastructure layers. Values are sourced from environment
variables, `streamlit` secrets or ``config.json`` via ``shared.config``.
"""
from __future__ import annotations

from typing import Dict

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
stub_max_runtime_warn: float = getattr(settings, "STUB_MAX_RUNTIME_WARN", 0.25)
STUB_MAX_RUNTIME_WARN: float = stub_max_runtime_warn

# Notification thresholds
risk_badge_threshold: float = getattr(settings, "RISK_BADGE_THRESHOLD", 0.75)
technical_signal_threshold: float = getattr(settings, "TECHNICAL_SIGNAL_THRESHOLD", 2)
earnings_upcoming_days: int = getattr(settings, "EARNINGS_UPCOMING_DAYS", 7)

# Backwards compatibility for legacy imports
YAHOO_FUNDAMENTALS_TTL: int = yahoo_fundamentals_ttl
YAHOO_QUOTES_TTL: int = yahoo_quotes_ttl
MIN_SCORE_THRESHOLD: int = min_score_threshold
MAX_RESULTS: int = max_results

# Feature flags
FEATURE_OPPORTUNITIES_TAB: bool = bool(
    getattr(settings, "FEATURE_OPPORTUNITIES_TAB", False)
)

# Macro data provider configuration
macro_api_provider: str = getattr(settings, "MACRO_API_PROVIDER", "fred")
fred_api_key: str | None = getattr(settings, "FRED_API_KEY", None)
fred_api_base_url: str = getattr(
    settings, "FRED_API_BASE_URL", "https://api.stlouisfed.org/fred"
)
fred_api_rate_limit_per_minute: int = getattr(
    settings, "FRED_API_RATE_LIMIT_PER_MINUTE", 120
)
fred_sector_series: Dict[str, str] = getattr(settings, "FRED_SECTOR_SERIES", {})
macro_sector_fallback: Dict[str, Dict[str, object]] = getattr(
    settings, "MACRO_SECTOR_FALLBACK", {}
)
world_bank_api_key: str | None = getattr(settings, "WORLD_BANK_API_KEY", None)
world_bank_api_base_url: str = getattr(
    settings, "WORLD_BANK_API_BASE_URL", "https://api.worldbank.org/v2"
)
world_bank_api_rate_limit_per_minute: int = getattr(
    settings, "WORLD_BANK_API_RATE_LIMIT_PER_MINUTE", 60
)
world_bank_sector_series: Dict[str, str] = getattr(
    settings, "WORLD_BANK_SECTOR_SERIES", {}
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
    "stub_max_runtime_warn",
    "STUB_MAX_RUNTIME_WARN",
    "YAHOO_FUNDAMENTALS_TTL",
    "YAHOO_QUOTES_TTL",
    "MIN_SCORE_THRESHOLD",
    "MAX_RESULTS",
    "risk_badge_threshold",
    "technical_signal_threshold",
    "earnings_upcoming_days",
    "FEATURE_OPPORTUNITIES_TAB",
    "macro_api_provider",
    "fred_api_key",
    "fred_api_base_url",
    "fred_api_rate_limit_per_minute",
    "fred_sector_series",
    "macro_sector_fallback",
    "world_bank_api_key",
    "world_bank_api_base_url",
    "world_bank_api_rate_limit_per_minute",
    "world_bank_sector_series",
]
