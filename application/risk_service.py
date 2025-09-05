from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "compute_returns",
    "annualized_volatility",
    "beta",
    "historical_var",
    "markowitz_optimize",
    "monte_carlo_simulation",
    "apply_stress",
]


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula retornos porcentuales diarios a partir de precios."""
    if price_df is None or price_df.empty:
        return pd.DataFrame()
    return price_df.pct_change().dropna(how="all")


def annualized_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    """Volatilidad anualizada de una serie de retornos diarios."""
    if returns is None or returns.empty:
        return float("nan")
    return returns.std(ddof=0) * np.sqrt(trading_days)


def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Beta del portafolio contra un benchmark."""
    if portfolio_returns.empty or benchmark_returns.empty:
        return float("nan")
    cov = np.cov(portfolio_returns, benchmark_returns)[0][1]
    var_b = np.var(benchmark_returns)
    if var_b == 0:
        return float("nan")
    return cov / var_b


def historical_var(portfolio_returns: pd.Series, alpha: float = 0.05) -> float:
    """Value-at-Risk histórico (percentil alpha)."""
    if portfolio_returns.empty:
        return float("nan")
    return portfolio_returns.quantile(alpha)


def markowitz_optimize(returns_df: pd.DataFrame) -> pd.Series:
    """Pesos de mínima varianza (Markowitz) normalizados."""
    if returns_df is None or returns_df.empty:
        return pd.Series(dtype="float64")
    cov = returns_df.cov().values
    inv_cov = np.linalg.pinv(cov)
    ones = np.ones(len(cov))
    w = inv_cov @ ones
    w /= ones @ inv_cov @ ones
    return pd.Series(w, index=returns_df.columns)


def monte_carlo_simulation(returns_df: pd.DataFrame, weights: pd.Series | np.ndarray,
                           n_sims: int = 1000, horizon: int = 30) -> np.ndarray:
    """Simulación Monte Carlo de retornos de cartera."""
    if returns_df is None or returns_df.empty:
        return np.array([])
    w = np.array(weights)
    mu = returns_df.mean().values
    cov = returns_df.cov().values
    sims = []
    for _ in range(int(n_sims)):
        path = np.random.multivariate_normal(mu, cov, size=int(horizon))
        port_ret = path @ w
        final_val = np.prod(1 + port_ret)
        sims.append(final_val)
    return np.array(sims)


def apply_stress(weights: pd.Series | np.ndarray, shocks: dict[str, float]) -> float:
    """Aplica shocks porcentuales a los pesos de la cartera y devuelve el cambio agregado."""
    if weights is None:
        return float("nan")
    w = pd.Series(weights)
    shock_series = pd.Series({k: shocks.get(k, 0.0) for k in w.index})
    stressed = w * (1 + shock_series)
    return stressed.sum()