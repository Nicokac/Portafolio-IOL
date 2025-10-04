"""Unit tests for newly added risk calculations."""
from __future__ import annotations

import sys
from pathlib import Path

import logging

import numpy as np
from numpy.random import SeedSequence
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

    mean = returns.mean().values
    cov = returns.cov().values
    w = weights.reindex(returns.columns).fillna(0.0).values
    expected_rng = np.random.default_rng(SeedSequence(123))
    sims = expected_rng.multivariate_normal(mean, cov, size=(n_sims, horizon))
    expected_paths = np.prod(1 + sims @ w, axis=1) - 1
    expected = pd.Series(expected_paths)

    result = rs.monte_carlo_simulation(
        returns,
        weights,
        n_sims=n_sims,
        horizon=horizon,
        rng=np.random.default_rng(SeedSequence(123)),
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

    first = rs.monte_carlo_simulation(
        returns,
        weights,
        n_sims=128,
        horizon=16,
        rng=np.random.default_rng(SeedSequence(2024)),
    )

    second = rs.monte_carlo_simulation(
        returns,
        weights,
        n_sims=128,
        horizon=16,
        rng=np.random.default_rng(SeedSequence(2024)),
    )

    pd.testing.assert_series_equal(first, second)

    third = rs.monte_carlo_simulation(
        returns,
        weights,
        n_sims=128,
        horizon=16,
        rng=np.random.default_rng(SeedSequence(2025)),
    )

    assert not third.equals(first)


def test_monte_carlo_simulation_batches_match_vectorised() -> None:
    """Batched sampling should produce the same paths as vectorised sampling."""

    returns = pd.DataFrame(
        {
            "AL30": [0.015, -0.01, 0.012, 0.02],
            "GGAL": [0.01, 0.008, -0.004, 0.018],
            "PAMP": [-0.002, 0.014, 0.006, 0.009],
        }
    )
    weights = pd.Series({"AL30": 0.4, "GGAL": 0.35, "PAMP": 0.25})

    seed = SeedSequence(321)
    vectorised = rs.monte_carlo_simulation(
        returns, weights, n_sims=4096, horizon=64, rng=np.random.default_rng(seed)
    )

    seed = SeedSequence(321)
    batched = rs.monte_carlo_simulation(
        returns,
        weights,
        n_sims=4096,
        horizon=64,
        rng=np.random.default_rng(seed),
        batch_size=512,
    )

    pd.testing.assert_series_equal(vectorised, batched)


@pytest.mark.slow
@pytest.mark.parametrize(
    "batch_size",
    [None, 16384],
    ids=["vectorised", "batched"],
)
def test_monte_carlo_simulation_large_sample_stability(batch_size: int | None) -> None:
    """Large Monte Carlo runs should remain statistically stable.

    Esta prueba verifica que, con n_sims=100k y un horizonte de 260 días,
    el error muestral se mantiene acotado. En CI admitimos tiempos de ejecución
    de hasta ~3 segundos; si se supera, considere ajustar ``batch_size`` o
    reducir temporalmente el escenario.
    """

    returns = pd.DataFrame(
        {
            "AL30": [0.012, -0.008, 0.01, 0.018, -0.004, 0.007],
            "GGAL": [0.009, 0.011, -0.006, 0.017, 0.003, -0.002],
            "YPFD": [0.008, -0.005, 0.013, 0.016, -0.007, 0.01],
        }
    )
    weights = pd.Series({"AL30": 0.45, "GGAL": 0.35, "YPFD": 0.20})

    n_sims = 100_000
    horizon = 260

    result = rs.monte_carlo_simulation(
        returns,
        weights,
        n_sims=n_sims,
        horizon=horizon,
        rng=np.random.default_rng(SeedSequence(2024)),
        batch_size=batch_size,
    )

    assert result.size == n_sims
    assert result.notna().all()

    sample_mean = float(result.mean())
    sample_std = float(result.std(ddof=0))
    assert 3.0 < sample_mean < 3.7
    assert 0.3 < sample_std < 0.6

    grouped = result.to_numpy().reshape(10, -1).mean(axis=1)
    assert grouped.std(ddof=0) < 6e-3


def test_apply_stress_combines_weights_and_shocks() -> None:
    """`apply_stress` should align indexes and apply percentage shocks."""

    prices = pd.Series({"AL30": 120.0, "GGAL": 80.0, "PAMP": 55.0})
    weights = pd.Series({"AL30": 0.5, "GGAL": 0.3, "PAMP": 0.2})
    shocks = {"AL30": -0.1, "GGAL": 0.05, "DLR": 0.2}

    stressed_prices = prices * pd.Series(shocks).reindex(prices.index).fillna(0.0).add(1)
    expected_value = float((stressed_prices * weights).sum())

    result = rs.apply_stress(prices, weights, shocks)

    assert result == pytest.approx(expected_value)


def test_markowitz_optimize_degrades_on_singular_covariance(caplog: pytest.LogCaptureFixture) -> None:
    """Singular covariance matrices should yield NaN weights instead of crashing."""

    # Identical assets force a singular covariance matrix
    returns = pd.DataFrame({
        "A": [0.01, 0.02, -0.01, 0.015],
        "B": [0.01, 0.02, -0.01, 0.015],
    })

    caplog.set_level(logging.WARNING)
    weights = rs.markowitz_optimize(returns)

    assert isinstance(weights, pd.Series)
    assert list(weights.index) == list(returns.columns)
    assert weights.isna().all()
    assert "covariance matrix is singular" in caplog.text


def test_markowitz_optimize_returns_nan_when_weights_not_normalisable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If the tangency weights cannot be normalised the result should degrade."""

    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.0, 0.01],
            "B": [0.03, -0.01, 0.02, 0.0],
        }
    )

    mean = returns.mean() * 252
    assert pytest.approx(mean["A"]) == pytest.approx(mean["B"])

    caplog.set_level(logging.WARNING)
    weights = rs.markowitz_optimize(returns, risk_free=float(mean["A"]))

    assert isinstance(weights, pd.Series)
    assert weights.index.tolist() == returns.columns.tolist()
    assert weights.isna().all()
    assert "invalid normalisation factor" in caplog.text


def test_monte_carlo_simulation_handles_invalid_covariance() -> None:
    """Monte Carlo should return a safe series when the covariance is invalid."""

    returns = pd.DataFrame({
        "A": [0.01, -0.02, 0.015],
        "B": [0.005, -0.01, 0.02],
    })
    weights = pd.Series({"A": 0.6, "B": 0.4})

    class RaisingRNG:
        def multivariate_normal(self, *args, **kwargs):
            raise np.linalg.LinAlgError("not positive definite")

    result = rs.monte_carlo_simulation(
        returns,
        weights,
        n_sims=128,
        horizon=16,
        rng=RaisingRNG(),
    )

    assert isinstance(result, pd.Series)
    assert result.size == 1
    assert result.isna().all()


def test_monte_carlo_simulation_detects_non_positive_covariance(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative eigenvalues should be caught before sampling to avoid crashes."""

    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03],
            "B": [0.015, 0.025, 0.035],
        }
    )
    weights = pd.Series({"A": 0.5, "B": 0.5})

    def fake_cov(self: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [[1.0, 0.8], [0.8, -0.2]], index=self.columns, columns=self.columns
        )

    monkeypatch.setattr(pd.DataFrame, "cov", fake_cov, raising=False)

    caplog.set_level(logging.WARNING)
    result = rs.monte_carlo_simulation(returns, weights, n_sims=64, horizon=8)

    assert isinstance(result, pd.Series)
    assert result.size == 1
    assert result.isna().all()
    assert "negative eigenvalues" in caplog.text
