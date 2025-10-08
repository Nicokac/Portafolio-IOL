"""Benchmark and factor analysis utilities for risk module."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

try:  # pragma: no cover - compatibility shim for SciPy >=1.16
    import scipy._lib._util as _scipy_util
except Exception:  # pragma: no cover - SciPy not available or altered
    _scipy_util = None
else:  # pragma: no cover - executed during runtime, hard to unit test deterministically
    if not hasattr(_scipy_util, "_lazywhere"):
        def _lazywhere(cond, arrays, func, fillvalue=np.nan):
            cond_arr = np.asarray(cond, dtype=bool)
            array_list = [np.asarray(arr) for arr in arrays]
            if not array_list:
                return np.array([], dtype=float)
            result = np.full_like(array_list[0], fillvalue, dtype=float)
            if cond_arr.any():
                result[cond_arr] = func(*[arr[cond_arr] for arr in array_list])
            return result

        _scipy_util._lazywhere = _lazywhere

import statsmodels.api as sm

__all__ = ["benchmark_analysis"]


def _to_series(data: pd.Series | Mapping | np.ndarray | list) -> pd.Series:
    if isinstance(data, pd.Series):
        series = data.copy()
    else:
        series = pd.Series(data)
    return series.astype(float).dropna()


def benchmark_analysis(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    factors_df: pd.DataFrame | None = None,
) -> dict[str, float | dict[str, float]]:
    """Compute tracking metrics and optional factor regression for a portfolio.

    Parameters
    ----------
    portfolio_returns:
        Daily returns for the portfolio.
    benchmark_returns:
        Daily returns for the selected benchmark.
    factors_df:
        Optional dataframe containing factor returns aligned by date.

    Returns
    -------
    dict
        Dictionary including tracking error, active return, information ratio
        and optional factor betas / R-squared from a multi-factor regression.
    """

    port = _to_series(portfolio_returns)
    bench = _to_series(benchmark_returns)

    aligned = pd.concat([port.rename("portfolio"), bench.rename("benchmark")], axis=1)
    aligned = aligned.dropna()

    if aligned.empty:
        tracking_error = float("nan")
        active_return = float("nan")
    else:
        diff = aligned["portfolio"] - aligned["benchmark"]
        tracking_error = float(np.std(diff) * np.sqrt(252))
        active_return = float(aligned["portfolio"].mean() - aligned["benchmark"].mean())

    information_ratio = float("nan")
    if not np.isnan(tracking_error) and tracking_error != 0.0:
        information_ratio = float(active_return / tracking_error)

    betas: dict[str, float] = {}
    r_squared = float("nan")

    if factors_df is not None:
        factors = pd.DataFrame(factors_df).dropna(how="all")
        if not factors.empty and not aligned.empty:
            combined = aligned.join(factors, how="inner").dropna()
            factor_cols = [col for col in combined.columns if col not in {"portfolio", "benchmark"}]
            if factor_cols:
                y = combined["portfolio"].astype(float)
                X = combined[factor_cols].astype(float)
                X = sm.add_constant(X, has_constant="add")
                model = sm.OLS(y, X).fit()
                betas = {
                    col: float(model.params.get(col, np.nan)) for col in factor_cols
                }
                r_squared = float(model.rsquared)

    return {
        "tracking_error": float(tracking_error) if tracking_error == tracking_error else np.nan,
        "active_return": float(active_return) if active_return == active_return else np.nan,
        "information_ratio": information_ratio if information_ratio == information_ratio else np.nan,
        "factor_betas": betas,
        "r_squared": r_squared if r_squared == r_squared else np.nan,
    }
