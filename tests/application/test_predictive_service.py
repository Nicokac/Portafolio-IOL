from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from application.predictive_service import (
    get_cache_stats,
    predict_sector_performance,
    reset_cache,
)


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


def test_predict_sector_performance_aggregates_correlations() -> None:
    reset_cache()
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

    result = predict_sector_performance(
        opportunities,
        backtesting_service=service,
        cache=DummyCache(),
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


def test_predict_sector_performance_uses_cache_and_stats() -> None:
    reset_cache()
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
    assert cache.ttl.get("sector_predictions") == pytest.approx(21600.0)
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
