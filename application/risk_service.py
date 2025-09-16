from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict

__all__ = [
    "compute_returns",
    "portfolio_returns",
    "annualized_volatility",
    "beta",
    "historical_var",
    "markowitz_optimize",
    "monte_carlo_simulation",
    "apply_stress",
]


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula rendimientos porcentuales a partir de precios."""
    if price_df is None or price_df.empty:
        return pd.DataFrame()
    return price_df.sort_index().pct_change().dropna(how="all")


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Calcula rendimientos de una cartera a partir de pesos y rendimientos."""
    if returns is None or returns.empty:
        return pd.Series(dtype=float)
    w = weights.reindex(returns.columns).fillna(0.0)
    return returns.dot(w)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Volatilidad anualizada de una serie de rendimientos."""
    if returns is None or len(returns) == 0:
        return 0.0
    return float(returns.std() * np.sqrt(periods_per_year))


def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Beta de la cartera respecto a un benchmark."""
    if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) == 0:
        return float("nan")
    cov = np.cov(portfolio_returns, benchmark_returns)
    return float(cov[0, 1] / cov[1, 1])


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Value at Risk (VaR) histórico para la confianza dada."""
    if returns is None or len(returns) == 0:
        return 0.0
    q = np.quantile(returns, 1 - confidence)
    return float(-q)


def markowitz_optimize(returns: pd.DataFrame, risk_free: float = 0.0) -> pd.Series:
    """Obtiene pesos del portafolio de tangencia según Markowitz."""
    if returns is None or returns.empty:
        return pd.Series(dtype=float)
    mean_ret = returns.mean() * 252
    cov = returns.cov() * 252
    inv_cov = np.linalg.inv(cov.values)
    ones = np.ones(len(mean_ret))
    excess = mean_ret.values - risk_free
    w = inv_cov @ excess
    w /= ones @ w
    return pd.Series(w, index=returns.columns)


def monte_carlo_simulation(
    returns: pd.DataFrame,
    weights: pd.Series,
    n_sims: int = 1000,
    horizon: int = 252,
) -> pd.Series:
    """Simula distribución de rendimientos de cartera mediante Monte Carlo."""
    if returns is None or returns.empty:
        return pd.Series(dtype=float)
    mean = returns.mean().values
    cov = returns.cov().values
    w = weights.reindex(returns.columns).fillna(0.0).values
    sims = np.random.multivariate_normal(mean, cov, size=(n_sims, horizon))
    port_paths = np.prod(1 + sims @ w, axis=1) - 1
    return pd.Series(port_paths)


def apply_stress(
    prices: pd.Series,
    weights: pd.Series,
    shocks: Dict[str, float],
) -> float:
    """Aplica shocks porcentuales a precios y calcula valor de la cartera."""
    if prices is None or len(prices) == 0:
        return 0.0
    w = weights.reindex(prices.index).fillna(0.0)
    s = pd.Series(shocks).reindex(prices.index).fillna(0.0)
    stressed = prices * (1 + s)
    return float((stressed * w).sum())
