"""Tests for the transitional backtesting service."""

from __future__ import annotations

import pandas as pd
import pytest

from application.backtesting_service import (
    BacktestingService,
    load_prices_from_fixture,
)
from services.cache import CacheService


def test_backtesting_service_runs_fixture_success() -> None:
    service = BacktestingService()
    result = service.run("TEST", strategy="sma")
    assert not result.empty
    assert "equity" in result.columns
    assert result["equity"].iloc[0] == pytest.approx(1.0, rel=1e-3)


def test_backtesting_service_uses_cache(monkeypatch) -> None:
    calls: list[str] = []

    def loader(symbol: str) -> pd.DataFrame:
        calls.append(symbol)
        return load_prices_from_fixture(symbol)

    cache = CacheService(namespace="test-backtest")
    service = BacktestingService(cache=cache, data_loader=loader)

    first = service.run("TEST")
    second = service.run("TEST")
    pd.testing.assert_frame_equal(first, second)
    assert calls == ["TEST"]


def test_cache_service_respects_ttl() -> None:
    clock = {"value": 0.0}

    def fake_monotonic() -> float:
        return clock["value"]

    cache = CacheService(namespace="ttl", monotonic=fake_monotonic)

    cache.set("alpha", "a", ttl=5)
    assert cache.get("alpha") == "a"

    clock["value"] += 6
    assert cache.get("alpha") is None

    call_count = {"value": 0}

    def loader() -> str:
        call_count["value"] += 1
        return "fresh"

    assert cache.get_or_set("beta", loader, ttl=5) == "fresh"
    assert cache.get_or_set("beta", loader, ttl=5) == "fresh"
    assert call_count["value"] == 1

    clock["value"] += 6
    assert cache.get_or_set("beta", loader, ttl=5) == "fresh"
    assert call_count["value"] == 2
