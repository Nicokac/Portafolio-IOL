"""Unit tests for newly added risk calculations."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from application.risk_service import drawdown_series


def test_drawdown_series_empty():
    result = drawdown_series(pd.Series(dtype=float))
    assert isinstance(result, pd.Series)
    assert result.empty


def test_drawdown_series_non_empty():
    returns = pd.Series([0.01, -0.02, 0.03])
    result = drawdown_series(returns)

    expected = pd.Series([0.0, -0.02, 0.0])

    assert not result.empty
    assert (result <= 0).all()
    pd.testing.assert_series_equal(
        result.reset_index(drop=True).round(8),
        expected.reset_index(drop=True).round(8),
        check_names=False,
    )
from application import risk_service as rs


def test_drawdown_matches_cumulative_equity() -> None:
    """The drawdown helper should follow the classical equity curve definition."""

    returns = pd.Series([0.05, -0.02, 0.03, -0.10, 0.04])
    dd = rs.drawdown(returns)

    equity = (1 + returns).cumprod()
    expected = equity / equity.cummax() - 1
    expected.name = "drawdown"

    pd.testing.assert_series_equal(dd, expected)
    assert rs.max_drawdown(returns) == pytest.approx(expected.min())


def test_drawdown_handles_empty_inputs() -> None:
    """Empty inputs should gracefully return empty series and neutral statistics."""

    empty = pd.Series(dtype=float)
    dd = rs.drawdown(empty)
    assert dd.empty
    assert rs.max_drawdown(empty) == 0.0


def test_beta_honours_min_periods() -> None:
    """`beta` should respect the configured minimum period requirement."""

    bench = pd.Series([0.01, 0.015, 0.012, -0.002])
    port = 1.5 * bench

    assert np.isnan(rs.beta(port, bench, min_periods=6))

    value = rs.beta(port, bench, min_periods=4)
    assert value == pytest.approx(1.5)


def test_beta_returns_nan_when_benchmark_variance_zero() -> None:
    """A flat benchmark should lead to an undefined beta (NaN)."""

    port = pd.Series([0.01, 0.02, -0.01])
    bench = pd.Series([0.0, 0.0, 0.0])

    assert np.isnan(rs.beta(port, bench))


def test_expected_shortfall_matches_tail_mean() -> None:
    """Expected shortfall should average the tail beyond the VaR threshold."""

    returns = pd.Series([0.02, -0.03, 0.01, -0.05, 0.04, -0.01])
    confidence = 0.95
    threshold = returns.quantile(1 - confidence)
    tail = returns[returns <= threshold]
    expected = float(-(tail.mean()))

    result = rs.expected_shortfall(returns, confidence=confidence)
    assert result == pytest.approx(expected)


def test_expected_shortfall_handles_empty_series() -> None:
    """Empty inputs should produce a neutral CVaR."""

    assert rs.expected_shortfall(pd.Series(dtype=float)) == 0.0


def test_rolling_correlations_computes_pairs() -> None:
    """Rolling correlations should return a column per pair."""

    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.DataFrame({
        "A": [0.01, 0.02, -0.01, 0.03, 0.00],
        "B": [0.00, 0.01, -0.02, 0.02, 0.01],
    }, index=idx)

    result = rs.rolling_correlations(returns, window=3)
    expected = returns["A"].rolling(3).corr(returns["B"])

    assert list(result.columns) == ["A↔B"]
    expected = expected.dropna()
    expected.name = "A↔B"
    pd.testing.assert_series_equal(
        result["A↔B"],
        expected,
    )


def test_rolling_correlations_requires_enough_assets() -> None:
    """Rolling correlations should bail out with insufficient data."""

    single = pd.DataFrame({"A": [0.01, 0.02, -0.01]})
    empty = rs.rolling_correlations(single, window=3)
    assert empty.empty


def test_monte_carlo_simulation_is_reproducible() -> None:
    """Monte Carlo simulations should honour the random seed and statistics."""

    returns = pd.DataFrame(
        {
            "AL30": [0.01, 0.015, -0.005, 0.02],
            "GGAL": [0.012, -0.01, 0.008, 0.015],
        }
    )
    weights = pd.Series({"AL30": 0.6, "GGAL": 0.4})

    n_sims = 256
    horizon = 32

    np.random.seed(123)
    mean = returns.mean().values
    cov = returns.cov().values
    w = weights.reindex(returns.columns).fillna(0.0).values
    sims = np.random.multivariate_normal(mean, cov, size=(n_sims, horizon))
    expected_paths = np.prod(1 + sims @ w, axis=1) - 1
    expected = pd.Series(expected_paths)

    np.random.seed(123)
    result = rs.monte_carlo_simulation(
        returns, weights, n_sims=n_sims, horizon=horizon
    )

    pd.testing.assert_series_equal(result, expected)
    assert result.mean() == pytest.approx(expected.mean())
    assert result.std() == pytest.approx(expected.std())


def test_monte_carlo_simulation_seed_control() -> None:
    """Resetting the seed should produce identical simulation paths."""

    returns = pd.DataFrame(
        {
            "AL30": [0.02, -0.01, 0.015],
            "GGAL": [-0.005, 0.012, 0.01],
        }
    )
    weights = pd.Series({"AL30": 0.5, "GGAL": 0.5})

    np.random.seed(2024)
    first = rs.monte_carlo_simulation(returns, weights, n_sims=128, horizon=16)

    np.random.seed(2024)
    second = rs.monte_carlo_simulation(returns, weights, n_sims=128, horizon=16)

    pd.testing.assert_series_equal(first, second)

    np.random.seed(2025)
    third = rs.monte_carlo_simulation(returns, weights, n_sims=128, horizon=16)

    assert not third.equals(first)


def test_apply_stress_combines_weights_and_shocks() -> None:
    """`apply_stress` should align indexes and apply percentage shocks."""

    prices = pd.Series({"AL30": 120.0, "GGAL": 80.0, "PAMP": 55.0})
    weights = pd.Series({"AL30": 0.5, "GGAL": 0.3, "PAMP": 0.2})
    shocks = {"AL30": -0.1, "GGAL": 0.05, "DLR": 0.2}

    stressed_prices = prices * pd.Series(shocks).reindex(prices.index).fillna(0.0).add(1)
    expected_value = float((stressed_prices * weights).sum())

    result = rs.apply_stress(prices, weights, shocks)

    assert result == pytest.approx(expected_value)
