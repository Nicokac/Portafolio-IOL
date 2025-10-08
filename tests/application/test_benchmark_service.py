import numpy as np
import pandas as pd
import pytest

from application.benchmark_service import benchmark_analysis


def test_tracking_error_and_active_return_alignment():
    portfolio = pd.Series([0.01, 0.015, 0.02, 0.005])
    benchmark = pd.Series([0.008, 0.011, 0.019, 0.004])

    result = benchmark_analysis(portfolio, benchmark)

    diff = (portfolio - benchmark).dropna()
    expected_te = np.std(diff) * np.sqrt(252)
    expected_active = diff.mean()

    assert pytest.approx(result["tracking_error"], rel=1e-6) == expected_te
    assert pytest.approx(result["active_return"], rel=1e-6) == expected_active


def test_information_ratio_matches_manual_calculation():
    portfolio = pd.Series([0.02, 0.01, 0.03, 0.015])
    benchmark = pd.Series([0.01, 0.01, 0.015, 0.012])

    result = benchmark_analysis(portfolio, benchmark)

    diff = (portfolio - benchmark).dropna()
    expected_te = np.std(diff) * np.sqrt(252)
    expected_ir = (portfolio.mean() - benchmark.mean()) / expected_te

    assert pytest.approx(result["information_ratio"], rel=1e-6) == expected_ir


def test_factor_regression_recovers_betas_and_r_squared():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    rate = pd.Series(np.linspace(0.01, 0.03, len(idx)), index=idx, name="tasa")
    inflation = pd.Series(np.linspace(0.005, 0.02, len(idx)), index=idx, name="inflacion")
    factors = pd.concat([rate, inflation], axis=1)

    portfolio = 0.6 * rate + 0.3 * inflation + 0.01
    benchmark = 0.45 * rate + 0.25 * inflation + 0.008

    result = benchmark_analysis(portfolio, benchmark, factors_df=factors)

    betas = result["factor_betas"]
    assert betas, "Expected betas to be computed"

    design = np.column_stack([
        np.ones(len(idx)),
        rate.to_numpy(),
        inflation.to_numpy(),
    ])
    expected_params = np.linalg.lstsq(design, portfolio.to_numpy(), rcond=None)[0]
    expected_betas = {
        "tasa": expected_params[1],
        "inflacion": expected_params[2],
    }

    for factor, expected in expected_betas.items():
        assert pytest.approx(betas[factor], rel=1e-6) == expected

    assert result["r_squared"] > 0.99
