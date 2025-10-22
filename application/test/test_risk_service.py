import numpy as np
import pandas as pd

from application import risk_service as rs


def _make_prices(rows: int = 100) -> pd.DataFrame:
    np.random.seed(0)
    dates = pd.date_range("2020-01-01", periods=rows)
    a = 100 * np.cumprod(1 + 0.01 + 0.02 * np.random.randn(rows))
    b = 80 * np.cumprod(1 + 0.005 + 0.015 * np.random.randn(rows))
    return pd.DataFrame({"AAA": a, "BBB": b}, index=dates)


def test_volatility_zero():
    r = pd.Series([0, 0, 0])
    vol = rs.annualized_volatility(r)
    assert vol == 0


def test_beta_two():
    bench = pd.Series([0.01, 0.02, -0.01])
    port = 2 * bench
    b = rs.beta(port, bench)
    assert np.isclose(b, 2.0)


def test_var_non_negative():
    r = pd.Series([-0.1, -0.05, 0.02, 0.03])
    var = rs.historical_var(r, confidence=0.95)
    assert var >= 0


def test_markowitz_weights_sum_one():
    prices = _make_prices()
    returns = rs.compute_returns(prices)
    w = rs.markowitz_optimize(returns)
    assert np.isclose(w.sum(), 1.0)


def test_monte_carlo_length():
    prices = _make_prices()
    returns = rs.compute_returns(prices)
    weights = pd.Series([0.5, 0.5], index=returns.columns)
    sims = rs.monte_carlo_simulation(returns, weights, n_sims=10, horizon=10)
    assert len(sims) == 10
