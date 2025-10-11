"""Adapters dedicated to sector level predictions."""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from predictive_engine.base import compute_sector_predictions
from predictive_engine.models import SectorPredictionSet

LOGGER = logging.getLogger(__name__)


def build_sector_prediction_frame(
    opportunities: pd.DataFrame | None,
    *,
    backtesting_service: object,
    run_backtest: Callable[[object, str], pd.DataFrame | None],
    extract_series: Callable[[pd.DataFrame, str], pd.Series],
    ema_predictor: Callable[[pd.Series, int], float | None],
    average_correlation: Callable[[dict[str, pd.Series]], pd.Series],
    span: int,
) -> pd.DataFrame:
    """Return a dataframe compatible with the legacy predictive service."""

    prediction_set: SectorPredictionSet = compute_sector_predictions(
        opportunities,
        backtesting_service=backtesting_service,
        run_backtest=run_backtest,
        extract_series=extract_series,
        ema_predictor=ema_predictor,
        average_correlation=average_correlation,
        span=span,
        logger=LOGGER,
    )
    return prediction_set.to_dataframe()
