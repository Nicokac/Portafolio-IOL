"""Predictive engine package consolidating statistical routines."""

from shared.version import __build_signature__ as _APP_BUILD_SIGNATURE
from shared.version import __version__ as _APP_VERSION

from .base import (
    calculate_adaptive_forecast,
    compute_sector_predictions,
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
    "__build_signature__",
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

__version__ = _APP_VERSION
__build_signature__ = _APP_BUILD_SIGNATURE
