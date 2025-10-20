"""Unified cache API exposing core primitives and specialised helpers."""

from __future__ import annotations

from . import quotes as _quotes
from . import ui_adapter as _ui_adapter
from .core import CacheService, PredictiveCacheState
from .fx_cache import fetch_fx_rates, get_fx_provider
from .portfolio_cache import fetch_portfolio
from .ratelimiter import RateLimiter
from .quotes import *  # noqa: F401,F403
from .ui_adapter import *  # noqa: F401,F403

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
