import pandas as pd
import numpy as np
import pytest

from services.cache import CacheService

from predictive_engine.adapters import run_adaptive_forecast
from predictive_engine.storage import load_forecast_history, save_forecast_history


def _sample_history() -> pd.DataFrame:
    timestamps = pd.date_range("2024-05-01", periods=4, freq="D")
    rows: list[dict[str, object]] = []
    for idx, ts in enumerate(timestamps):
        rows.append(
            {
                "timestamp": ts,
                "sector": "Energy",
                "predicted_return": 6.0 + idx * 0.2,
                "actual_return": 5.5 + idx * 0.15,
            }
        )
        rows.append(
            {
                "timestamp": ts,
                "sector": "Finance",
                "predicted_return": 4.0 + idx * 0.1,
                "actual_return": 3.6 + idx * 0.12,
            }
        )
    return pd.DataFrame(rows)


def test_beta_shift_and_dispersion_metrics(tmp_path):
    history = _sample_history()
    cache = CacheService(namespace="test_adaptive_metrics")
    history_path = tmp_path / "history.parquet"

    result = run_adaptive_forecast(
        history=history,
        cache=cache,
        ema_span=3,
        rolling_window=3,
        ttl_hours=0.1,
        max_history_rows=720,
        persist_state=False,
        persist_history=False,
        history_path=history_path,
        warm_start=False,
        state_key="state",
        correlation_key="corr",
        performance_prefix="test",
    )

    forecast = result["forecast"]
    steps = forecast.steps
    adjustments = (
        steps.pivot_table(
            index="timestamp",
            columns="sector",
            values="beta_adjustment",
            aggfunc="mean",
        )
        .sort_index()
        .fillna(0.0)
    )
    beta_diffs = adjustments.diff().abs().dropna(how="all")
    expected_beta_shift = 0.0
    if not beta_diffs.empty:
        expected_beta_shift = beta_diffs.mean(axis=1, skipna=True).dropna().mean()

    assert np.isfinite(forecast.metrics.beta_shift_avg)
    assert forecast.metrics.beta_shift_avg == pytest.approx(expected_beta_shift or 0.0, rel=1e-6)

    sector_dispersion_expected = (
        history.groupby("sector")["predicted_return"].mean().std(ddof=0)
    )
    assert forecast.metrics.sector_dispersion == pytest.approx(sector_dispersion_expected, rel=1e-6)


def test_storage_persistence_roundtrip(tmp_path):
    history = _sample_history()
    path = tmp_path / "adaptive_history.parquet"

    saved_path = save_forecast_history(history, path)
    assert saved_path.exists()

    loaded = load_forecast_history(saved_path)
    pd.testing.assert_frame_equal(
        history.sort_index(axis=1).reset_index(drop=True),
        loaded.sort_index(axis=1).reset_index(drop=True),
    )


def test_warm_start_loads_persisted_history(tmp_path):
    history = _sample_history()
    cache = CacheService(namespace="test_warm_start")
    history_path = tmp_path / "persisted.parquet"

    run_adaptive_forecast(
        history=history,
        cache=cache,
        ema_span=2,
        rolling_window=2,
        ttl_hours=0.1,
        max_history_rows=50,
        persist_state=True,
        persist_history=True,
        history_path=history_path,
        warm_start=True,
        state_key="state",
        correlation_key="corr",
        performance_prefix="warm",
    )

    cache.clear()

    new_predictions = pd.DataFrame(
        [
            {"sector": "Energy", "predicted_return": 6.5, "timestamp": "2024-05-10"},
            {"sector": "Finance", "predicted_return": 4.3, "timestamp": "2024-05-10"},
        ]
    )
    new_actuals = pd.DataFrame(
        [
            {"sector": "Energy", "actual_return": 6.0, "timestamp": "2024-05-10"},
            {"sector": "Finance", "actual_return": 4.0, "timestamp": "2024-05-10"},
        ]
    )

    update_result = run_adaptive_forecast(
        predictions=new_predictions,
        actuals=new_actuals,
        cache=cache,
        ema_span=2,
        ttl_hours=0.1,
        max_history_rows=50,
        persist_state=True,
        persist_history=False,
        history_path=history_path,
        warm_start=True,
        state_key="state",
        correlation_key="corr",
        performance_prefix="warm",
    )

    adaptive_update = update_result["update"]
    assert adaptive_update.state.history.shape[0] > 2
    assert cache.get("state") is not None
