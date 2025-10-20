"""Compatibility facade exposing cache services and adapters."""

from __future__ import annotations

import importlib

from services.cache.core import CacheService, PredictiveCacheState
from services.cache.fx_cache import fetch_fx_rates, get_fx_provider
from services.cache.portfolio_cache import fetch_portfolio
from services.cache.ratelimiter import RateLimiter
from services.cache.quotes import *  # noqa: F401,F403
from services.cache.ui_adapter import *  # noqa: F401,F403

_quotes = importlib.import_module("services.cache.quotes")
_ui_adapter = importlib.import_module("services.cache.ui_adapter")

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
)
