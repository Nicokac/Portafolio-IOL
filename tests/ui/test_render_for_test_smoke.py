"""Smoke test for the recommendations tab offline render helper."""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import pandas as pd
import pytest

from controllers.recommendations_controller import (
    AdaptiveForecastViewModel,
    PredictiveCacheViewModel,
    SectorPerformanceViewModel,
)
from ui.tabs import recommendations


class _ChartStub:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def update_traces(self, **kwargs):  # type: ignore[override]
        return self

    def update_layout(self, **kwargs):  # type: ignore[override]
        return self


_SUPPORTS_WARN_NONE: bool | None = None


@contextmanager
def _no_warnings():
    """Ensure compatibility with pytest.warns(None) across pytest versions."""

    global _SUPPORTS_WARN_NONE
    if _SUPPORTS_WARN_NONE is None:
        try:
            with pytest.warns(None):
                pass
        except TypeError:
            _SUPPORTS_WARN_NONE = False
        else:
            _SUPPORTS_WARN_NONE = True

    if _SUPPORTS_WARN_NONE:
        with pytest.warns(None):
            yield
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            yield


def test_render_for_test_smoke() -> None:
    """Ensure `_render_for_test` runs without warnings and hydrates session state."""

    dataset_path = Path("docs/fixtures/default/recommendations_sample.csv")
    df = pd.read_csv(dataset_path).head(3)
    state = SimpleNamespace(selected_mode="diversify")

    chart_stub = SimpleNamespace(
        pie=lambda *args, **kwargs: _ChartStub("pie"),
        bar=lambda *args, **kwargs: _ChartStub("bar"),
    )
    stub_history = pd.DataFrame(
        {
            "timestamp": ["2024-01-01"],
            "symbol": ["TEST"],
            "sector": ["Tecnología"],
            "value": [1.0],
        }
    )
    stub_summary = {
        "beta_mean": 1.0,
        "correlation_mean": 0.5,
        "sector_dispersion": 0.2,
        "beta_shift_avg": 0.1,
        "mae": 0.02,
        "rmse": 0.03,
        "bias": 0.01,
        "raw_mae": 0.05,
        "raw_rmse": 0.06,
        "raw_bias": 0.04,
    }
    stub_steps = pd.DataFrame({"timestamp": ["2024-01-01"], "mae": [0.02]})
    stub_cache = {"hit_ratio": 100.0, "last_updated": "offline"}

    original_px = recommendations.px
    original_prepare = recommendations.prepare_adaptive_history
    original_generate = recommendations.generate_synthetic_history
    original_sector_view = recommendations.recommendations_controller.load_sector_performance_view
    original_simulate = recommendations.recommendations_controller.run_adaptive_forecast_view
    original_export = recommendations.export_adaptive_report
    original_build = recommendations.build_correlation_figure
    original_stats = recommendations.recommendations_controller.get_predictive_cache_view

    start = perf_counter()
    try:
        recommendations.px = chart_stub
        recommendations.prepare_adaptive_history = lambda frame: stub_history
        recommendations.generate_synthetic_history = lambda frame: stub_history
        recommendations.recommendations_controller.load_sector_performance_view = (
            lambda frame, **_: SectorPerformanceViewModel(
                predictions=frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(),
                cache=PredictiveCacheViewModel(hits=1, misses=0, last_updated="offline"),
            )
        )
        recommendations.recommendations_controller.run_adaptive_forecast_view = (
            lambda history, **kwargs: AdaptiveForecastViewModel(
                payload={
                    "summary": dict(stub_summary),
                    "historical_correlation": pd.DataFrame(),
                    "rolling_correlation": pd.DataFrame(),
                    "correlation_matrix": pd.DataFrame([[1.0]]),
                    "beta_shift": pd.Series([0.1]),
                    "cache_metadata": dict(stub_cache),
                    "steps": stub_steps.copy(),
                },
                summary=dict(stub_summary),
                cache_metadata=dict(stub_cache),
            )
        )
        recommendations.export_adaptive_report = lambda payload: Path("docs/qa/v0.5.6-smoke-report.md")
        recommendations.build_correlation_figure = lambda *args, **kwargs: _ChartStub("correlation")
        recommendations.recommendations_controller.get_predictive_cache_view = lambda: PredictiveCacheViewModel(
            hits=3, misses=0, last_updated="offline"
        )

        with _no_warnings():
            recommendations._render_for_test(df, state)
    finally:
        recommendations.px = original_px
        recommendations.prepare_adaptive_history = original_prepare
        recommendations.generate_synthetic_history = original_generate
        recommendations.recommendations_controller.load_sector_performance_view = original_sector_view
        recommendations.recommendations_controller.run_adaptive_forecast_view = original_simulate
        recommendations.export_adaptive_report = original_export
        recommendations.build_correlation_figure = original_build
        recommendations.recommendations_controller.get_predictive_cache_view = original_stats
    duration = perf_counter() - start

    logging.getLogger("qa.smoke").info("render_for_test duration=%.3fs", duration)

    assert "_recommendations_state" in recommendations.st.session_state
    assert "recommendations" in recommendations.st.session_state["_recommendations_state"]
