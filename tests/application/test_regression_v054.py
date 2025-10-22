from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from application.adaptive_predictive_service import (
    export_adaptive_report,
    prepare_adaptive_history,
    simulate_adaptive_forecast,
)
from application.backtesting_service import BacktestingService, FixturePriceLoader
from application.predictive_service import reset_cache
from services.cache.core import CacheService

FIXTURES_ROOT = Path("docs/fixtures/default")


def _load_recommendations() -> pd.DataFrame:
    csv_path = FIXTURES_ROOT / "recommendations_sample.csv"
    frame = pd.read_csv(csv_path)
    if "sector" not in frame.columns:
        frame["sector"] = ["Technology", "Utilities", "Consumer Cyclical"][: len(frame)]
    return frame


def test_regression_adaptive_forecast_v054() -> None:
    reset_cache()
    loader = FixturePriceLoader(fixtures_root=FIXTURES_ROOT)
    backtesting = BacktestingService(data_loader=loader)
    recommendations = _load_recommendations()

    opportunities = recommendations[["symbol", "sector"]].copy()
    history = prepare_adaptive_history(opportunities, backtesting_service=backtesting)
    assert not history.empty, "Se espera historial construido a partir de fixtures"

    cache = CacheService(namespace="adaptive_regression_test")
    results = simulate_adaptive_forecast(history, cache=cache, persist=True)

    steps = results.get("steps")
    assert isinstance(steps, pd.DataFrame) and not steps.empty

    for column in ("raw_prediction", "adjusted_prediction", "actual_return"):
        series = pd.to_numeric(steps[column], errors="coerce")
        assert np.isfinite(series.sum())
        assert np.isfinite(series.mean())
        assert np.isfinite(series.std(ddof=0))

    assert results.get("mae", np.inf) < results.get("raw_mae", np.inf)

    cache_metadata = results.get("cache_metadata", {})
    assert cache_metadata.get("hit_ratio", 0.0) >= 45.0

    report_path = export_adaptive_report(results)
    assert report_path.exists()
    try:
        content = report_path.read_text(encoding="utf-8")
    finally:
        report_path.unlink(missing_ok=True)
        if report_path.parent.exists() and not any(report_path.parent.iterdir()):
            report_path.parent.rmdir()

    assert "Reporte adaptativo" in content
    assert "MAE adaptativo" in content
    assert "β-shift" in content
