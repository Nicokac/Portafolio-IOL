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

BENCHMARK_BASELINES: dict[str, dict[str, object]] = {
    "merval": {
        "name": "Merval",
        "expected_return": 14.5,
        "beta": 1.18,
        "weights": [0.45, 0.35, 0.2],
    },
    "sp500": {
        "name": "S&P 500",
        "expected_return": 9.5,
        "beta": 1.0,
        "weights": [0.34, 0.33, 0.33],
    },
    "bonos": {
        "name": "Bonos soberanos",
        "expected_return": 6.0,
        "beta": 0.65,
        "weights": [0.5, 0.3, 0.2],
    },
}

__all__ = ["benchmark_analysis", "compute_benchmark_comparison", "BENCHMARK_BASELINES"]


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
                betas = {col: float(model.params.get(col, np.nan)) for col in factor_cols}
                r_squared = float(model.rsquared)

    return {
        "tracking_error": float(tracking_error) if tracking_error == tracking_error else np.nan,
        "active_return": float(active_return) if active_return == active_return else np.nan,
        "information_ratio": information_ratio if information_ratio == information_ratio else np.nan,
        "factor_betas": betas,
        "r_squared": r_squared if r_squared == r_squared else np.nan,
    }


def compute_benchmark_comparison(recommendations: pd.DataFrame, benchmark: str) -> dict[str, float | str]:
    """Estimate relative metrics against a reference benchmark."""

    benchmark_key = str(benchmark).lower()
    baseline = BENCHMARK_BASELINES.get(benchmark_key)
    if baseline is None:
        return {
            "benchmark": benchmark,
            "label": str(benchmark).upper(),
            "portfolio_return": float("nan"),
            "benchmark_return": float("nan"),
            "relative_return": float("nan"),
            "portfolio_beta": float("nan"),
            "benchmark_beta": float("nan"),
            "relative_beta": float("nan"),
            "tracking_error": float("nan"),
        }

    frame = recommendations if isinstance(recommendations, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        weights = np.array([], dtype=float)
        expected = np.array([], dtype=float)
        betas = np.array([], dtype=float)
    else:
        weights = pd.to_numeric(frame.get("allocation_%"), errors="coerce").to_numpy()
        if not np.isfinite(weights).any() or float(np.nansum(weights)) <= 0:
            weights = np.repeat(1 / len(frame), len(frame)) if len(frame) else np.array([], dtype=float)
        else:
            weights = np.nan_to_num(weights) / float(np.nansum(weights))
        expected = pd.to_numeric(frame.get("expected_return"), errors="coerce").to_numpy()
        betas = pd.to_numeric(frame.get("beta"), errors="coerce").to_numpy()

    if weights.size and expected.size:
        portfolio_return = float(np.nansum(weights * np.nan_to_num(expected)))
    else:
        portfolio_return = float("nan")

    if weights.size and betas.size:
        portfolio_beta = float(np.nansum(weights * np.nan_to_num(betas)))
    else:
        portfolio_beta = float("nan")

    benchmark_return = float(baseline.get("expected_return", float("nan")))
    benchmark_beta = float(baseline.get("beta", float("nan")))

    relative_return = (
        portfolio_return - benchmark_return
        if np.isfinite(portfolio_return) and np.isfinite(benchmark_return)
        else float("nan")
    )
    relative_beta = (
        portfolio_beta - benchmark_beta if np.isfinite(portfolio_beta) and np.isfinite(benchmark_beta) else float("nan")
    )

    baseline_weights = np.array(baseline.get("weights", []), dtype=float)
    if baseline_weights.size == 0 and weights.size:
        baseline_weights = np.repeat(1 / weights.size, weights.size)
    if baseline_weights.size and weights.size:
        if baseline_weights.size != weights.size:
            repeats = int(np.ceil(weights.size / baseline_weights.size))
            baseline_weights = np.tile(baseline_weights, repeats)[: weights.size]
        if baseline_weights.sum() <= 0:
            baseline_weights = np.repeat(1 / weights.size, weights.size)
        else:
            baseline_weights = baseline_weights / float(baseline_weights.sum())
        tracking_error = float(np.sqrt(np.mean((weights - baseline_weights) ** 2)) * 100)
    else:
        tracking_error = float("nan")

    return {
        "benchmark": benchmark_key,
        "label": str(baseline.get("name", benchmark_key)).strip() or benchmark_key.upper(),
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
        "relative_return": relative_return,
        "portfolio_beta": portfolio_beta,
        "benchmark_beta": benchmark_beta,
        "relative_beta": relative_beta,
        "tracking_error": tracking_error,
    }
