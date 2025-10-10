"""Shared primitives for predictive analytics services."""

from .normalization import normalise_symbol_sector
from .state import PredictiveCacheState
from .forecast import (
    average_correlation,
    compute_ema_prediction,
    extract_backtest_series,
    run_backtest,
)

__all__ = [
    "normalise_symbol_sector",
    "PredictiveCacheState",
    "average_correlation",
    "compute_ema_prediction",
    "extract_backtest_series",
    "run_backtest",
]
