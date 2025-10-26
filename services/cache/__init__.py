"""Unified cache API exposing core primitives and specialised helpers."""

from __future__ import annotations

import logging

logging.getLogger(__name__).warning(
    "services.cache compatibility shim loaded; remove after 0.9.1"
)

from . import quotes as _quotes
from . import ui_adapter as _ui_adapter
from .core import CacheService, PredictiveCacheState  # noqa: F401
from .fx_cache import fetch_fx_rates, get_fx_provider  # noqa: F401
from .portfolio_cache import fetch_portfolio  # noqa: F401
from .quotes import *  # noqa: F401,F403
from .ratelimiter import RateLimiter  # noqa: F401
from .ui_adapter import *  # noqa: F401,F403

# Compat layer for legacy imports (to be removed in v0.9.1)
try:  # pragma: no cover - defensive import for compatibility
    import streamlit as st  # type: ignore[import-not-found]
    from infrastructure.iol.auth import IOLAuth  # type: ignore[import-not-found]
    from shared.telemetry import record_fx_api_response
except Exception:  # pragma: no cover - ensure shim availability even if deps missing
    st = None
    IOLAuth = None
    record_fx_api_response = lambda *a, **k: None

_COMPAT_EXPORTS = {"st", "IOLAuth", "record_fx_api_response"}

__all__ = sorted(
    {
        "CacheService",
        "PredictiveCacheState",
        "RateLimiter",
        "fetch_fx_rates",
        "fetch_portfolio",
        "get_fx_provider",
    }
    | set(getattr(_quotes, "__all__", ()))
    | set(getattr(_ui_adapter, "__all__", ()))
    | _COMPAT_EXPORTS
)
