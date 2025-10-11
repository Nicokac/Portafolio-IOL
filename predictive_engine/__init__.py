"""Predictive engine package consolidating statistical routines."""

from .base import (
    compute_sector_predictions,
    calculate_adaptive_forecast,
    evaluate_model_metrics,
)
from .models import (
    AdaptiveForecastResult,
    AdaptiveState,
    AdaptiveUpdateResult,
    CorrelationBundle,
    ModelMetrics,
    SectorPrediction,
    SectorPredictionSet,
)

__all__ = [
    "__version__",
    "compute_sector_predictions",
    "calculate_adaptive_forecast",
    "evaluate_model_metrics",
    "AdaptiveForecastResult",
    "AdaptiveState",
    "AdaptiveUpdateResult",
    "CorrelationBundle",
    "ModelMetrics",
    "SectorPrediction",
    "SectorPredictionSet",
]

__version__ = "0.6.3-part2"
