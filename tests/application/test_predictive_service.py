from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import logging

import application.predictive_service as predictive_service
from application.predictive_core.state import PredictiveCacheState
from application.predictive_service import (
    build_adaptive_history,
    get_cache_stats,
    predict_sector_performance,
)
from shared.settings import PREDICTIVE_TTL_HOURS


@pytest.fixture()
def cache_state(monkeypatch) -> PredictiveCacheState:
    predictive_service._CACHE.clear()
    state = PredictiveCacheState()
    monkeypatch.setattr(predictive_service, "_CACHE_STATE", state)
    return state


def _make_backtest(returns: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"strategy_ret": returns})
    frame["signal"] = 1
    frame["ret"] = returns
    frame["equity"] = (1 + returns).cumprod()
    return frame


class DummyBacktestingService:
    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data
        self.calls: list[str] = []

    def run(self, symbol: str, strategy: str = "sma") -> pd.DataFrame:  # noqa: D401
        self.calls.append(symbol)
        return self._data[str(symbol)].copy()


class DummyCache:
    def __init__(self) -> None:
        self.store: dict[str, pd.DataFrame] = {}
        self.ttl: dict[str, float | None] = {}

    def get(self, key: str, default: object = None) -> object:
        return self.store.get(key, default)

    def set(self, key: str, value: pd.DataFrame, *, ttl: float | None = None) -> pd.DataFrame:
        self.store[key] = value
        self.ttl[key] = ttl
        return value


def _last_cache_entry(cache: DummyCache) -> tuple[str, float | None]:
    assert cache.ttl, "Expected cache to contain entries"
    last_key = list(cache.ttl.keys())[-1]
    return last_key, cache.ttl[last_key]


def test_predict_sector_performance_aggregates_correlations(cache_state: PredictiveCacheState) -> None:
    returns_a = pd.Series([0.01, 0.012, 0.014, 0.015])
    returns_b = pd.Series([0.009, 0.011, 0.013, 0.014])
    returns_c = pd.Series([0.004, 0.006, 0.007, 0.008])
    service = DummyBacktestingService(
        {
            "AAA": _make_backtest(returns_a),
            "BBB": _make_backtest(returns_b),
            "CCC": _make_backtest(returns_c),
        }
    )
    opportunities = pd.DataFrame(
        [
            {"symbol": "AAA", "sector": "Technology"},
            {"symbol": "BBB", "sector": "Technology"},
            {"symbol": "CCC", "sector": "Utilities"},
        ]
    )

    dummy_cache = DummyCache()
    result = predict_sector_performance(
        opportunities,
        backtesting_service=service,
        cache=dummy_cache,
        span=2,
    )

    assert set(result["sector"]) == {"Technology", "Utilities"}
    tech_row = result.set_index("sector").loc["Technology"]
    util_row = result.set_index("sector").loc["Utilities"]

    ema_a = returns_a.ewm(span=2, adjust=False).mean().iloc[-1] * 100
    ema_b = returns_b.ewm(span=2, adjust=False).mean().iloc[-1] * 100
    expected_tech = np.mean([ema_a, ema_b])

    assert tech_row["sample_size"] == 2
    assert tech_row["predicted_return"] == pytest.approx(expected_tech, rel=1e-6)
    assert 0.0 <= tech_row["confidence"] <= 1.0
    assert util_row["sample_size"] == 1
    assert util_row["avg_correlation"] == pytest.approx(0.0, abs=1e-6)
    assert cache_state.misses == 1
    assert cache_state.hits == 0
    cache_key, _ = _last_cache_entry(dummy_cache)
    assert "AAA" in cache_key and "CCC" in cache_key


def test_predict_sector_performance_uses_cache_and_stats(cache_state: PredictiveCacheState) -> None:
    returns = pd.Series([0.01, 0.011, 0.012, 0.014])
    data = {"AAA": _make_backtest(returns)}
    service = DummyBacktestingService(data)
    cache = DummyCache()
    opportunities = pd.DataFrame(
        [
            {"symbol": "AAA", "sector": "Healthcare"},
        ]
    )

    first = predict_sector_performance(
        opportunities,
        backtesting_service=service,
        cache=cache,
        span=3,
    )
    stats_after_first = get_cache_stats()
    assert stats_after_first.misses == 1
    expected_ttl = PREDICTIVE_TTL_HOURS * 3600.0
    _, stored_ttl = _last_cache_entry(cache)
    assert stored_ttl == pytest.approx(expected_ttl)
    assert service.calls == ["AAA"]

    second = predict_sector_performance(
        opportunities,
        backtesting_service=service,
        cache=cache,
        span=3,
    )

    stats_after_second = get_cache_stats()
    assert stats_after_second.hits >= 1
    assert len(service.calls) == 1  # cache hit avoids new calls
    assert first.equals(second)
    assert stats_after_second.hit_ratio > 0
    assert cache_state.misses == 1
    assert cache_state.hits == 1


def test_predict_sector_performance_accepts_custom_ttl(cache_state: PredictiveCacheState) -> None:
    returns = pd.Series([0.01, 0.012, 0.013])
    data = {"AAA": _make_backtest(returns)}
    service = DummyBacktestingService(data)
    cache = DummyCache()
    opportunities = pd.DataFrame([
        {"symbol": "AAA", "sector": "Technology"},
    ])

    predict_sector_performance(
        opportunities,
        backtesting_service=service,
        cache=cache,
        span=2,
        ttl_hours=0.01,
    )

    _, stored_ttl = _last_cache_entry(cache)
    assert stored_ttl == pytest.approx(36.0)
    assert cache_state.ttl_hours == pytest.approx(0.01)


def test_predictive_ttl_respects_monkeypatch(monkeypatch, cache_state: PredictiveCacheState) -> None:
    monkeypatch.setattr(predictive_service, "PREDICTIVE_TTL_HOURS", 0.02)
    predictive_service.reset_cache()
    monkeypatch.setattr(predictive_service, "_CACHE_STATE", cache_state)

    returns = pd.Series([0.02, 0.021, 0.022])
    data = {"AAA": _make_backtest(returns)}
    service = DummyBacktestingService(data)
    cache = DummyCache()
    opportunities = pd.DataFrame([
        {"symbol": "AAA", "sector": "Finance"},
    ])

    predict_sector_performance(
        opportunities,
        backtesting_service=service,
        cache=cache,
        span=2,
    )

    _, stored_ttl = _last_cache_entry(cache)
    assert stored_ttl == pytest.approx(0.02 * 3600.0)
    assert cache_state.ttl_hours == pytest.approx(0.02)


def test_build_adaptive_history_real_mode_uses_cache() -> None:
    returns_a = pd.Series([0.01, 0.012, 0.013])
    returns_b = pd.Series([0.008, 0.007, 0.009])
    service = DummyBacktestingService(
        {
            "AAA": _make_backtest(returns_a),
            "BBB": _make_backtest(returns_b),
        }
    )
    cache = DummyCache()
    opportunities = pd.DataFrame(
        [
            {"symbol": "AAA", "sector": "Tech"},
            {"symbol": "BBB", "sector": "Finance"},
        ]
    )

    history_first = build_adaptive_history(
        opportunities,
        mode="real",
        backtesting_service=service,
        span=2,
        max_symbols=2,
        cache=cache,
    )

    assert not history_first.empty
    assert len(service.calls) == 2

    history_second = build_adaptive_history(
        opportunities,
        mode="real",
        backtesting_service=service,
        span=2,
        max_symbols=2,
        cache=cache,
    )

    assert history_second.equals(history_first)
    assert len(service.calls) == 2


def test_build_adaptive_history_synthetic_clips_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    recommendations = pd.DataFrame(
        [
            {"symbol": "AAA", "sector": "Tech", "predicted_return_pct": 0.75},
            {"symbol": "BBB", "sector": "Finance", "predicted_return_pct": -1.2},
        ]
    )

    with caplog.at_level(logging.WARNING, logger="application.predictive_service"):
        history = build_adaptive_history(recommendations, mode="synthetic", periods=3)

    assert not history.empty
    assert history["predicted_return"].between(-60, 60).all()
    assert any("truncando" in record.message for record in caplog.records)
