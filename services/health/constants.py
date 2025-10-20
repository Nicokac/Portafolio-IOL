"""Shared constants used by health metrics helpers."""

from __future__ import annotations

_HEALTH_KEY = "health_metrics"
_MARKET_DATA_INCIDENTS_KEY = "market_data_incidents"
_MARKET_DATA_INCIDENT_LIMIT = 20
_LATENCY_FAST_THRESHOLD_MS = 250.0
_LATENCY_MEDIUM_THRESHOLD_MS = 750.0
_PROVIDER_HISTORY_LIMIT = 8
_TAB_LATENCIES_KEY = "tab_latencies"
_TAB_LATENCY_HISTORY_LIMIT = 32
_ADAPTER_FALLBACK_KEY = "adapter_fallbacks"
_RISK_INCIDENTS_KEY = "risk_incidents"
_RISK_INCIDENT_HISTORY_LIMIT = 50
_QUOTE_RATE_LIMIT_KEY = "quote_rate_limits"
_SNAPSHOT_EVENT_KEY = "snapshot_event"
_ENVIRONMENT_SNAPSHOT_KEY = "environment_snapshot"
_DIAGNOSTICS_SNAPSHOT_KEY = "startup_diagnostics"
_PORTFOLIO_HISTORY_LIMIT = 32
_QUOTE_HISTORY_LIMIT = 32
_FX_API_HISTORY_LIMIT = 32
_FX_CACHE_HISTORY_LIMIT = 32
_SESSION_MONITORING_KEY = "session_monitoring"
_ACTIVE_SESSIONS_KEY = "active_sessions"
_LOGIN_TO_RENDER_STATS_KEY = "login_to_render"
_LAST_HTTP_ERROR_KEY = "last_http_error"
_SESSION_MONITORING_TTL_SECONDS = 300.0
_DEPENDENCIES_KEY = "dependencies"

_PROVIDER_LABELS = {
    "alpha_vantage": "Alpha Vantage",
    "av": "Alpha Vantage",
    "polygon": "Polygon",
    "fred": "FRED",
    "worldbank": "World Bank",
    "iol": "IOL v2",
    "legacy": "IOL Legacy",
    "stale": "Cache persistente",
    "cache": "Caché en memoria",
}

_QUOTE_PROVIDER_HISTORY_LIMIT = 12

__all__ = [
    "_HEALTH_KEY",
    "_MARKET_DATA_INCIDENTS_KEY",
    "_MARKET_DATA_INCIDENT_LIMIT",
    "_LATENCY_FAST_THRESHOLD_MS",
    "_LATENCY_MEDIUM_THRESHOLD_MS",
    "_PROVIDER_HISTORY_LIMIT",
    "_TAB_LATENCIES_KEY",
    "_TAB_LATENCY_HISTORY_LIMIT",
    "_ADAPTER_FALLBACK_KEY",
    "_RISK_INCIDENTS_KEY",
    "_RISK_INCIDENT_HISTORY_LIMIT",
    "_QUOTE_RATE_LIMIT_KEY",
    "_SNAPSHOT_EVENT_KEY",
    "_ENVIRONMENT_SNAPSHOT_KEY",
    "_DIAGNOSTICS_SNAPSHOT_KEY",
    "_PORTFOLIO_HISTORY_LIMIT",
    "_QUOTE_HISTORY_LIMIT",
    "_FX_API_HISTORY_LIMIT",
    "_FX_CACHE_HISTORY_LIMIT",
    "_SESSION_MONITORING_KEY",
    "_ACTIVE_SESSIONS_KEY",
    "_LOGIN_TO_RENDER_STATS_KEY",
    "_LAST_HTTP_ERROR_KEY",
    "_SESSION_MONITORING_TTL_SECONDS",
    "_DEPENDENCIES_KEY",
    "_PROVIDER_LABELS",
    "_QUOTE_PROVIDER_HISTORY_LIMIT",
]
