"""Servicios de aplicación para el ecosistema Portafolio-IOL."""

from shared.version import __build_signature__ as _APP_BUILD_SIGNATURE
from shared.version import __version__ as _APP_VERSION

from .adaptive_predictive_service import (
    export_adaptive_report,
    prepare_adaptive_history,
    simulate_adaptive_forecast,
)
from .predictive_service import build_adaptive_history
from .backtesting_service import BacktestingService
from .portfolio_service import PortfolioService
from .predictive_service import get_cache_stats, predict_sector_performance, reset_cache
from .recommendation_service import RecommendationService

__version__ = _APP_VERSION
__build_signature__ = _APP_BUILD_SIGNATURE

__all__ = [
    "BacktestingService",
    "PortfolioService",
    "RecommendationService",
    "export_adaptive_report",
    "build_adaptive_history",
    "prepare_adaptive_history",
    "simulate_adaptive_forecast",
    "predict_sector_performance",
    "get_cache_stats",
    "reset_cache",
    "__version__",
    "__build_signature__",
]
