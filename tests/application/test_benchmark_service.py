import math

import pandas as pd
import pytest

from application.benchmark_service import (
    BENCHMARK_BASELINES,
    compute_benchmark_comparison,
)


def test_compute_benchmark_comparison_returns_relative_metrics() -> None:
    recommendations = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "allocation_%": [40.0, 35.0, 25.0],
            "expected_return": [12.0, 8.0, 6.0],
            "beta": [1.1, 0.9, 0.8],
        }
    )

    metrics = compute_benchmark_comparison(recommendations, "sp500")

    assert metrics["benchmark"] == "sp500"
    assert metrics["label"] == BENCHMARK_BASELINES["sp500"]["name"]
    assert metrics["portfolio_return"] == pytest.approx(9.1, rel=1e-3)
    expected_relative = 9.1 - BENCHMARK_BASELINES["sp500"]["expected_return"]
    assert metrics["relative_return"] == pytest.approx(expected_relative, rel=1e-3)
    assert metrics["portfolio_beta"] == pytest.approx(0.955, rel=1e-3)
    assert metrics["relative_beta"] == pytest.approx(0.955 - BENCHMARK_BASELINES["sp500"]["beta"], rel=1e-3)
    assert metrics["tracking_error"] > 0


def test_compute_benchmark_comparison_handles_unknown_index() -> None:
    df = pd.DataFrame()
    metrics = compute_benchmark_comparison(df, "unknown")

    assert metrics["benchmark"] == "unknown"
    assert math.isnan(metrics["relative_return"])
    assert math.isnan(metrics["tracking_error"])
