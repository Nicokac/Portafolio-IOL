"""Shared forecasting utilities for predictive services."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

def run_backtest(
    service: object,
    symbol: str,
    *,
    strategy: str = "sma",
    logger: logging.Logger | None = None,
) -> pd.DataFrame | None:
    """Execute ``service.run`` defensively and return a DataFrame or ``None``."""

    if service is None:
        return None
    try:
        run = getattr(service, "run")
    except AttributeError:
        return None
    try:
        backtest = run(symbol, strategy=strategy)
    except Exception:  # pragma: no cover - defensive logging
        if logger is not None:
            logger.exception("Falló la ejecución de backtesting para %s", symbol)
        return None
    if not isinstance(backtest, pd.DataFrame) or backtest.empty:
        return None
    return backtest


def extract_backtest_series(backtest: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series from ``column`` or an empty series if unavailable."""

    if not isinstance(backtest, pd.DataFrame) or column not in backtest.columns:
        return pd.Series(dtype=float)
    series = pd.to_numeric(backtest[column], errors="coerce").dropna()
    if not isinstance(series, pd.Series):  # pragma: no cover - safety
        return pd.Series(dtype=float)
    return series


def compute_ema_prediction(returns: pd.Series, *, span: int) -> float | None:
    """Compute the latest EMA-based prediction in percentage points."""

    if not isinstance(returns, pd.Series) or returns.empty:
        return None
    span = max(int(span), 1)
    ema = returns.ewm(span=span, adjust=False).mean()
    if ema.empty:
        return None
    return float(ema.iloc[-1]) * 100.0


def average_correlation(symbol_returns: dict[str, pd.Series]) -> pd.Series:
    """Return the average correlation for each symbol in ``symbol_returns``."""

    if not symbol_returns:
        return pd.Series(dtype=float)
    aligned = pd.DataFrame(symbol_returns).dropna(how="all")
    if aligned.empty:
        return pd.Series(0.0, index=list(symbol_returns))
    if aligned.shape[1] < 2:
        return pd.Series(0.0, index=list(aligned.columns))
    corr_matrix = aligned.corr().replace([np.inf, -np.inf], np.nan)
    np.fill_diagonal(corr_matrix.values, np.nan)
    return corr_matrix.mean(axis=1, skipna=True).fillna(0.0)


__all__ = [
    "average_correlation",
    "compute_ema_prediction",
    "extract_backtest_series",
    "run_backtest",
]
