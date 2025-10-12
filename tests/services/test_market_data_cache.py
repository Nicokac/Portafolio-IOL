from __future__ import annotations

from typing import Callable

import pandas as pd
import pytest

from services.cache.core import CacheService
from services.cache.market_data_cache import MarketDataCache


class FakeClock:
    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += float(seconds)


@pytest.fixture()
def market_cache() -> MarketDataCache:
    clock = FakeClock()
    history = CacheService(namespace="test_history", monotonic=clock)
    fundamentals = CacheService(namespace="test_fund", monotonic=clock)
    cache = MarketDataCache(
        history_cache=history,
        fundamentals_cache=fundamentals,
        default_ttl=2.0,
    )
    cache._clock = clock  # type: ignore[attr-defined]
    return cache


def _call_counting_loader(df: pd.DataFrame) -> Callable[[], pd.DataFrame]:
    calls = {"count": 0}

    def loader() -> pd.DataFrame:
        calls["count"] += 1
        return df

    loader.calls = calls  # type: ignore[attr-defined]
    return loader


def test_history_cache_returns_clone_and_reuses_value(market_cache: MarketDataCache):
    df = pd.DataFrame({"close": [1.0, 2.0]})
    loader = _call_counting_loader(df)

    first = market_cache.get_history(["AAPL"], loader=loader, period="6mo")
    assert loader.calls["count"] == 1
    first.loc[:, "close"] = 0.0

    second = market_cache.get_history(["AAPL"], loader=loader, period="6mo")
    assert loader.calls["count"] == 1
    pd.testing.assert_series_equal(second["close"], pd.Series([1.0, 2.0], name="close"))


def test_history_cache_expires_after_ttl(monkeypatch, market_cache: MarketDataCache):
    clock = getattr(market_cache, "_clock", None)
    assert clock is not None

    df = pd.DataFrame({"close": [5.0, 6.0]})
    loader = _call_counting_loader(df)

    _ = market_cache.get_history(["MSFT"], loader=loader, period="1y")
    assert loader.calls["count"] == 1

    clock.advance(1.0)
    _ = market_cache.get_history(["MSFT"], loader=loader, period="1y")
    assert loader.calls["count"] == 1

    clock.advance(2.1)
    _ = market_cache.get_history(["MSFT"], loader=loader, period="1y")
    assert loader.calls["count"] == 2


def test_fundamentals_cache_uses_symbol_and_sector_keys(market_cache: MarketDataCache):
    df = pd.DataFrame({"symbol": ["AAPL"], "metric": [1.0]})
    loader = _call_counting_loader(df)

    market_cache.get_fundamentals(["AAPL"], loader=loader, sectors=["Tech"])
    assert loader.calls["count"] == 1

    market_cache.get_fundamentals(["AAPL"], loader=loader, sectors=["Tech"])
    assert loader.calls["count"] == 1

    market_cache.get_fundamentals(["AAPL"], loader=loader, sectors=["Finance"])
    assert loader.calls["count"] == 2


def test_persistent_cache_survives_reinitialisation(tmp_path, monkeypatch):
    from services.cache import market_data_cache as module

    backend = module._SQLiteBackend(tmp_path / "market_cache.db")
    monkeypatch.setattr(module, "_BACKEND", backend)
    monkeypatch.setattr(module, "_initialise_backend", lambda: backend)

    cache_one = module.create_persistent_cache("test_persistent")
    cache_one.set("alpha", {"value": 1}, ttl=10.0)

    cache_two = module.create_persistent_cache("test_persistent")
    stored = cache_two.get("alpha")
    assert stored == {"value": 1}
