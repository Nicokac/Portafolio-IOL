from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from application.adaptive_predictive_service import (  # noqa: E402
    AdaptiveModelState,
    _CORR_KEY,
    _STATE_KEY,
    simulate_adaptive_forecast,
    update_model,
)
from services.cache import CacheService  # noqa: E402


class TrackingCache(CacheService):
    def __init__(self) -> None:
        self._now = 1_000.0
        super().__init__(namespace="test_adaptive", monotonic=self._tick)
        self.ttl_map: dict[str, float | None] = {}

    def _tick(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += float(seconds)

    def set(self, key: str, value, *, ttl: float | None = None):  # type: ignore[override]
        self.ttl_map[self._full_key(key)] = ttl
        return super().set(key, value, ttl=ttl)


def _build_frame(predicted_a: float, predicted_b: float, actual_a: float, actual_b: float, *, ts: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.DataFrame(
        [
            {"sector": "Technology", "predicted_return": predicted_a, "timestamp": ts},
            {"sector": "Finance", "predicted_return": predicted_b, "timestamp": ts},
        ]
    )
    actuals = pd.DataFrame(
        [
            {"sector": "Technology", "actual_return": actual_a, "timestamp": ts},
            {"sector": "Finance", "actual_return": actual_b, "timestamp": ts},
        ]
    )
    return predictions, actuals


def test_update_model_applies_ema_and_persists_state() -> None:
    cache = TrackingCache()

    preds1, actuals1 = _build_frame(10.0, 6.0, 5.0, 6.0, ts="2024-01-01")
    result1 = update_model(preds1, actuals1, cache=cache, ema_span=2, persist=True)

    assert isinstance(result1["history"], pd.DataFrame)
    assert not result1["history"].empty
    assert "Technology" in result1["beta_shift"].index
    assert result1["beta_shift"].loc["Technology"] == pytest.approx(-1.0, rel=1e-6)

    preds2, actuals2 = _build_frame(4.0, 5.5, 5.0, 5.4, ts="2024-01-02")
    result2 = update_model(preds2, actuals2, cache=cache, ema_span=2, persist=True)

    shift_second = result2["beta_shift"].loc["Technology"]
    assert shift_second == pytest.approx(-0.2, rel=1e-6)

    full_state_key = cache._full_key(_STATE_KEY)
    full_corr_key = cache._full_key(_CORR_KEY)
    assert cache.ttl_map[full_state_key] == pytest.approx(43_200.0)
    assert cache.ttl_map[full_corr_key] == pytest.approx(43_200.0)

    cached_state = cache.get(_STATE_KEY)
    assert isinstance(cached_state, AdaptiveModelState)
    assert len(cached_state.history) == 4


def test_simulate_adaptive_forecast_reduces_error() -> None:
    timestamps = pd.date_range("2024-02-01", periods=5, freq="D")
    rows = []
    for idx, ts in enumerate(timestamps):
        rows.extend(
            [
                {
                    "timestamp": ts,
                    "sector": "Technology",
                    "predicted_return": 8.0,
                    "actual_return": 6.0 + (idx % 2) * 0.3,
                },
                {
                    "timestamp": ts,
                    "sector": "Finance",
                    "predicted_return": 5.5,
                    "actual_return": 4.5 + (idx % 3) * 0.2,
                },
            ]
        )
    history = pd.DataFrame(rows)

    cache = TrackingCache()
    result = simulate_adaptive_forecast(history, ema_span=3, cache=cache, persist=False)

    assert result["raw_mae"] > result["mae"]
    assert result["raw_rmse"] > result["rmse"]
    assert isinstance(result["beta_shift"], pd.Series)
    assert not result["beta_shift"].empty
    assert result["summary"]["beta_mean"] == pytest.approx(result["beta_shift"].mean(), rel=1e-6)


def test_state_persists_across_simulation_and_updates() -> None:
    timestamps = pd.date_range("2024-03-01", periods=3, freq="D")
    history = pd.DataFrame(
        {
            "timestamp": np.tile(timestamps, 2),
            "sector": ["Energy", "Utilities"] * 3,
            "predicted_return": [7.0, 4.2, 6.5, 4.1, 6.8, 4.3],
            "actual_return": [6.0, 3.8, 6.2, 3.6, 6.1, 3.7],
        }
    )

    cache = TrackingCache()
    simulate_adaptive_forecast(history, ema_span=2, cache=cache, persist=True)

    state_after_sim = cache.get(_STATE_KEY)
    assert isinstance(state_after_sim, AdaptiveModelState)
    initial_len = len(state_after_sim.history)
    assert initial_len > 0

    preds, actuals = _build_frame(5.5, 4.0, 5.0, 3.7, ts="2024-03-10")
    update_model(preds, actuals, cache=cache, ema_span=2, persist=True)

    updated_state = cache.get(_STATE_KEY)
    assert isinstance(updated_state, AdaptiveModelState)
    assert len(updated_state.history) > initial_len
    assert cache.ttl_map[cache._full_key(_STATE_KEY)] == pytest.approx(43_200.0)

