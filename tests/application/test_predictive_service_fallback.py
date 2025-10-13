"""Regression tests for predictive_service degraded cache mode."""

from __future__ import annotations

import builtins
import importlib
import sys


def test_lazy_cache_fallback_when_market_cache_missing(monkeypatch):
    """Predictive service should provide a fallback cache if the dependency is absent."""

    monkeypatch.delitem(sys.modules, "application.predictive_service", raising=False)
    monkeypatch.delitem(sys.modules, "services.cache.market_data_cache", raising=False)

    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("services.cache.market_data_cache"):
            raise ImportError("market data cache missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    predictive_service = importlib.import_module("application.predictive_service")

    fallback = predictive_service.lazy_get_cache()
    assert isinstance(fallback, predictive_service.FallbackMarketDataCache)

    cache = fallback.prediction_cache
    assert cache.get("foo") is None
    cache.set("foo", "bar")  # should be a no-op without raising
    cache.set_ttl_override(60)
    assert cache.status() == {"backend": "fallback", "available": False}

    key = fallback.build_prediction_key(["AAA"], span=5, sectors=["Tech"])
    assert "fallback" in key

    monkeypatch.delitem(sys.modules, "application.predictive_service", raising=False)
