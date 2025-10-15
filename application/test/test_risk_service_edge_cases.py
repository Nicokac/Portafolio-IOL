import numpy as np
import pandas as pd

from application.risk_service import (
    annualized_volatility,
    beta,
    compute_returns,
    historical_var,
    markowitz_optimize,
    monte_carlo_simulation,
    portfolio_returns,
)


def test_compute_returns_empty_df():
    df = pd.DataFrame()
    result = compute_returns(df)
    assert result.empty


def test_compute_returns_with_data():
    prices = pd.DataFrame({"A": [100, 110], "B": [200, 220]}, index=pd.date_range("2020-01-01", periods=2))
    result = compute_returns(prices)
    expected = pd.DataFrame({"A": [0.1], "B": [0.1]}, index=pd.DatetimeIndex([pd.Timestamp("2020-01-02")]))
    pd.testing.assert_frame_equal(result, expected, check_freq=False)


def test_portfolio_returns_empty_df():
    returns = pd.DataFrame()
    weights = pd.Series(dtype=float)
    result = portfolio_returns(returns, weights)
    assert result.empty


def test_portfolio_returns_with_data():
    returns = pd.DataFrame({"A": [0.1, 0.2], "B": [0.0, 0.1]})
    weights = pd.Series({"A": 0.6, "B": 0.4})
    result = portfolio_returns(returns, weights)
    expected = pd.Series([0.1 * 0.6 + 0.0 * 0.4, 0.2 * 0.6 + 0.1 * 0.4])
    pd.testing.assert_series_equal(result, expected)


def test_annualized_volatility_empty_series():
    r = pd.Series(dtype=float)
    vol = annualized_volatility(r)
    assert vol == 0.0


def test_beta_mismatched_lengths():
    a = pd.Series([0.1, 0.2])
    b = pd.Series([0.1, 0.2, 0.3])
    result = beta(a, b)
    assert np.isnan(result)


def test_beta_empty_series():
    a = pd.Series(dtype=float)
    b = pd.Series(dtype=float)
    result = beta(a, b)
    assert np.isnan(result)


def test_historical_var_none_series():
    assert historical_var(None) == 0.0


def test_markowitz_optimize_empty_df():
    returns = pd.DataFrame()
    result = markowitz_optimize(returns)
    assert result.empty


def test_monte_carlo_simulation_empty_df():
    returns = pd.DataFrame()
    weights = pd.Series(dtype=float)
    result = monte_carlo_simulation(returns, weights)
    assert result.empty


