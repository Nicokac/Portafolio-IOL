from __future__ import annotations

from itertools import combinations
import logging

import numpy as np
from numpy.random import SeedSequence
import pandas as pd


LOGGER = logging.getLogger(__name__)

default_rng: np.random.Generator = np.random.default_rng(SeedSequence())

__all__ = [
    "default_rng",
    "compute_returns",
    "portfolio_returns",
    "annualized_volatility",
    "beta",
    "drawdown",
    "drawdown_series",
    "max_drawdown",
    "historical_var",
    "expected_shortfall",
    "rolling_correlations",
    "markowitz_optimize",
    "monte_carlo_simulation",
    "asset_risk_breakdown",
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

def beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    min_periods: int | None = None,
) -> float:
    """Beta de la cartera respecto a un benchmark configurable."""

    if portfolio_returns is None or benchmark_returns is None:
        return float("nan")

    port = pd.Series(portfolio_returns).dropna()
    bench = pd.Series(benchmark_returns).dropna()

    if len(port) != len(bench) or len(port) == 0:
        return float("nan")

    if min_periods is not None and (
        len(port) < int(min_periods) or len(bench) < int(min_periods)
    ):
        return float("nan")

    cov = np.cov(port, bench)
    denom = cov[1, 1]
    if denom == 0:
        return float("nan")
    return float(cov[0, 1] / denom)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown series (in percentage terms) from returns."""

    if returns is None:
        return pd.Series(dtype=float)

    series = pd.Series(returns)

    if series.empty:
        return pd.Series(dtype=float)

    cumulative = (1.0 + series.fillna(0.0)).cumprod()
    peaks = cumulative.cummax()
    drawdowns = cumulative.subtract(peaks).divide(peaks)
    return drawdowns


def drawdown(returns: pd.Series) -> pd.Series:
    """Serie de *drawdown* acumulado a partir de rendimientos porcentuales."""

    if returns is None or len(returns) == 0:
        return pd.Series(dtype=float)

    series = pd.Series(returns).fillna(0.0)
    equity = (1 + series).cumprod()
    running_max = equity.cummax()
    dd = equity.divide(running_max).subtract(1.0)
    dd.name = "drawdown"
    return dd


def max_drawdown(returns: pd.Series) -> float:
    """Retorna el *maximum drawdown* (valor mínimo de la serie de drawdown)."""

    if returns is None or len(returns) == 0:
        return 0.0

    dd = drawdown(returns)
    if dd.empty:
        return 0.0
    return float(dd.min())


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Value at Risk (VaR) histórico para la confianza dada."""
    if returns is None or len(returns) == 0:
        return 0.0
    q = np.quantile(returns, 1 - confidence)
    return float(-q)


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR / Expected Shortfall for the given confidence level."""

    if returns is None or len(returns) == 0:
        return 0.0

    series = pd.Series(returns).dropna()
    if series.empty:
        return 0.0

    threshold = series.quantile(1 - confidence)
    tail = series[series <= threshold]
    if tail.empty:
        return 0.0

    return float(-tail.mean())


def rolling_correlations(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute rolling correlations between all asset pairs for a given window."""

    if returns is None or returns.empty or window <= 1:
        return pd.DataFrame()

    df = returns.sort_index().dropna(how="all")
    if df.empty or df.shape[1] < 2:
        return pd.DataFrame()

    window = int(window)
    result = {}
    for a, b in combinations(df.columns, 2):
        pair_key = f"{a}↔{b}"
        result[pair_key] = df[a].rolling(window).corr(df[b])

    if not result:
        return pd.DataFrame()

    rcorr = pd.DataFrame(result)
    rcorr = rcorr.dropna(how="all")
    return rcorr


def asset_risk_breakdown(returns: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return annualised volatility and max drawdown per asset."""
    if returns is None or returns.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    vols = returns.std().fillna(0.0) * np.sqrt(252)
    cumulative = (1 + returns.fillna(0.0)).cumprod()
    peaks = cumulative.cummax()
    drawdowns = cumulative.divide(peaks).sub(1.0)
    max_dd = drawdowns.min().fillna(0.0)
    return vols, max_dd


def markowitz_optimize(returns: pd.DataFrame, risk_free: float = 0.0) -> pd.Series:
    """Obtiene pesos del portafolio de tangencia según Markowitz."""
    if returns is None or returns.empty:
        return pd.Series(dtype=float)
    mean_ret = returns.mean() * 252
    cov = returns.cov() * 252

    cov_values = cov.values
    if cov_values.size == 0:
        return pd.Series(dtype=float)

    if not np.all(np.isfinite(cov_values)):
        LOGGER.warning("markowitz_optimize: covariance contains non-finite values")
        return pd.Series(np.nan, index=returns.columns)

    rank = np.linalg.matrix_rank(cov_values)
    if rank < cov_values.shape[0]:
        LOGGER.warning("markowitz_optimize: covariance matrix is singular (rank %s < %s)", rank, cov_values.shape[0])
        return pd.Series(np.nan, index=returns.columns)

    try:
        inv_cov = np.linalg.pinv(cov_values)
    except (np.linalg.LinAlgError, ValueError):
        LOGGER.exception("markowitz_optimize: failed to compute pseudoinverse")
        return pd.Series(np.nan, index=returns.columns)

    ones = np.ones(len(mean_ret))
    excess = mean_ret.values - risk_free
    w = inv_cov @ excess
    denom = ones @ w
    if not np.isfinite(denom) or np.isclose(denom, 0.0, atol=1e-12, rtol=1e-12):
        LOGGER.warning("markowitz_optimize: invalid normalisation factor for weights")
        return pd.Series(np.nan, index=returns.columns)

    w /= denom

    if not np.all(np.isfinite(w)):
        LOGGER.warning("markowitz_optimize: resulting weights contain non-finite values")
        return pd.Series(np.nan, index=returns.columns)

    return pd.Series(w, index=returns.columns)


def monte_carlo_simulation(
    returns: pd.DataFrame,
    weights: pd.Series,
    n_sims: int = 1000,
    horizon: int = 252,
    *,
    rng: np.random.Generator | None = None,
    batch_size: int | None = None,
) -> pd.Series:
    """Simula distribución de rendimientos de cartera mediante Monte Carlo.

    Parameters
    ----------
    returns, weights
        Historicos de rendimientos y pesos de la cartera.
    n_sims, horizon
        Número de simulaciones y horizonte temporal de cada trayectoria.
    rng
        Generador opcional de NumPy para controlar la semilla y reproducibilidad.
    batch_size
        Cuando se especifica, permite generar las simulaciones en lotes para
        reducir la huella de memoria con valores grandes de ``n_sims``.
    """
    if returns is None or returns.empty:
        return pd.Series(dtype=float)
    mean = returns.mean().values
    cov = returns.cov().values

    if cov.size == 0:
        return pd.Series(dtype=float)

    if not np.all(np.isfinite(cov)):
        LOGGER.warning("monte_carlo_simulation: covariance contains non-finite values")
        return pd.Series(np.nan)

    if cov.shape[0] != cov.shape[1]:
        LOGGER.warning("monte_carlo_simulation: covariance matrix is not square")
        return pd.Series(np.nan)

    try:
        eigenvalues = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        LOGGER.warning("monte_carlo_simulation: unable to compute eigenvalues of covariance")
        return pd.Series(np.nan)

    if np.any(eigenvalues < -np.finfo(float).eps):
        LOGGER.warning(
            "monte_carlo_simulation: covariance matrix has negative eigenvalues (min=%s)",
            eigenvalues.min(),
        )
        return pd.Series(np.nan)

    w = weights.reindex(returns.columns).fillna(0.0).values

    def _sample(size: int) -> np.ndarray | None:
        try:
            generator = default_rng if rng is None else rng
            return generator.multivariate_normal(mean, cov, size=(size, horizon))
        except (np.linalg.LinAlgError, ValueError):
            return None

    if batch_size is not None:
        try:
            batch = int(batch_size)
        except (TypeError, ValueError):
            batch = None
        if batch is not None and batch > 0 and batch < n_sims:
            results: list[np.ndarray] = []
            remaining = int(n_sims)
            while remaining > 0:
                current = min(batch, remaining)
                sims = _sample(current)
                if sims is None or sims.ndim != 3:
                    return pd.Series(np.nan)
                results.append(np.prod(1 + sims @ w, axis=1) - 1)
                remaining -= current
            return pd.Series(np.concatenate(results))

    sims = _sample(int(n_sims))
    if sims is None or sims.ndim != 3:
        return pd.Series(np.nan)
    port_paths = np.prod(1 + sims @ w, axis=1) - 1
    return pd.Series(port_paths)
