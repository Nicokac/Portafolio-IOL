"""Bridges between the predictive engine and legacy application services."""

from .sector_adapter import build_sector_prediction_frame
from .forecast_adapter import (
    EngineUpdateContext,
    build_adaptive_updater,
    update_model_with_cache,
)

__all__ = [
    "build_sector_prediction_frame",
    "EngineUpdateContext",
    "build_adaptive_updater",
    "update_model_with_cache",
]
