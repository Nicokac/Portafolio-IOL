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
fastapi_tokens_key: str | None = getattr(settings, "fastapi_tokens_key", None)
app_env: str = getattr(settings, "app_env", "dev")
market_data_cache_backend: str = getattr(settings, "MARKET_DATA_CACHE_BACKEND", "sqlite")
market_data_cache_path: str | None = getattr(settings, "MARKET_DATA_CACHE_PATH", None)
market_data_cache_redis_url: str | None = getattr(settings, "MARKET_DATA_CACHE_REDIS_URL", None)
market_data_cache_ttl: float = getattr(settings, "MARKET_DATA_CACHE_TTL", 6 * 60 * 60)
redis_url: str | None = getattr(settings, "REDIS_URL", None)
enable_prometheus: bool = getattr(settings, "ENABLE_PROMETHEUS", True)
performance_verbose_text_log: bool = getattr(settings, "PERFORMANCE_VERBOSE_TEXT_LOG", False)
yahoo_request_delay: float = getattr(settings, "YAHOO_REQUEST_DELAY", 0.0)
quotes_ttl_seconds: int = getattr(settings, "QUOTES_TTL_SECONDS", 300)
quotes_rps_iol: float = getattr(settings, "QUOTES_RPS_IOL", 3.0)
quotes_rps_legacy: float = getattr(settings, "QUOTES_RPS_LEGACY", 1.0)
legacy_login_max_retries: int = getattr(settings, "LEGACY_LOGIN_MAX_RETRIES", 1)
legacy_login_backoff_base: float = getattr(settings, "LEGACY_LOGIN_BACKOFF_BASE", 0.5)
notifications_url: str | None = getattr(settings, "NOTIFICATIONS_URL", None)
notifications_timeout: float = getattr(settings, "NOTIFICATIONS_TIMEOUT", 3.0)
min_score_threshold: int = settings.min_score_threshold
max_results: int = settings.max_results
log_retention_days: int = getattr(settings, "LOG_RETENTION_DAYS", 7)
stub_max_runtime_warn: float = getattr(settings, "STUB_MAX_RUNTIME_WARN", 0.25)
STUB_MAX_RUNTIME_WARN: float = stub_max_runtime_warn

CACHE_HIT_THRESHOLDS: dict[str, float] = {"green": 0.7, "yellow": 0.4}

PREDICTIVE_TTL_HOURS: float = getattr(settings, "PREDICTIVE_TTL_HOURS", 6.0)
ADAPTIVE_TTL_HOURS: float = getattr(settings, "ADAPTIVE_TTL_HOURS", 12.0)

# Snapshot storage configuration
snapshot_backend: str = getattr(settings, "snapshot_backend", "json")
snapshot_storage_path: str | None = getattr(settings, "snapshot_storage_path", None)
snapshot_retention: int | None = getattr(settings, "snapshot_retention", None)

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
fmp_api_key: str | None = getattr(settings, "FMP_API_KEY", None)
fmp_base_url: str = getattr(
    settings, "FMP_BASE_URL", "https://financialmodelingprep.com/api/v3"
)
fmp_timeout: float = getattr(settings, "FMP_TIMEOUT", 5.0)
ohlc_primary_provider: str = getattr(settings, "OHLC_PRIMARY_PROVIDER", "alpha_vantage")
ohlc_secondary_providers: list[str] = list(
    getattr(settings, "OHLC_SECONDARY_PROVIDERS", []) or []
)
alpha_vantage_api_key: str | None = getattr(settings, "ALPHA_VANTAGE_API_KEY", None)
alpha_vantage_base_url: str = getattr(
    settings, "ALPHA_VANTAGE_BASE_URL", "https://www.alphavantage.co/query"
)
polygon_api_key: str | None = getattr(settings, "POLYGON_API_KEY", None)
polygon_base_url: str = getattr(settings, "POLYGON_BASE_URL", "https://api.polygon.io")

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
    "market_data_cache_backend",
    "market_data_cache_path",
    "market_data_cache_redis_url",
    "redis_url",
    "enable_prometheus",
    "performance_verbose_text_log",
    "market_data_cache_ttl",
    "yahoo_request_delay",
    "quotes_ttl_seconds",
    "quotes_rps_iol",
    "quotes_rps_legacy",
    "legacy_login_max_retries",
    "legacy_login_backoff_base",
    "notifications_url",
    "notifications_timeout",
    "min_score_threshold",
    "max_results",
    "log_retention_days",
    "stub_max_runtime_warn",
    "STUB_MAX_RUNTIME_WARN",
    "CACHE_HIT_THRESHOLDS",
    "PREDICTIVE_TTL_HOURS",
    "ADAPTIVE_TTL_HOURS",
    "snapshot_backend",
    "snapshot_storage_path",
    "snapshot_retention",
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
    "fmp_api_key",
    "fmp_base_url",
    "fmp_timeout",
    "ohlc_primary_provider",
    "ohlc_secondary_providers",
    "alpha_vantage_api_key",
    "alpha_vantage_base_url",
    "polygon_api_key",
    "polygon_base_url",
    "fastapi_tokens_key",
    "app_env",
]
