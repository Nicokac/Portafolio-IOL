"""Bridges between the predictive engine and legacy application services."""

from .forecast_adapter import (
    EngineUpdateContext,
    build_adaptive_updater,
    run_adaptive_forecast,
    update_model_with_cache,
)
from .sector_adapter import build_sector_prediction_frame

__all__ = [
    "build_sector_prediction_frame",
    "EngineUpdateContext",
    "build_adaptive_updater",
    "run_adaptive_forecast",
    "update_model_with_cache",
]
