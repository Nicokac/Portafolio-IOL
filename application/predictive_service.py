"""Sector-level predictive analytics with adaptive caching."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from application.backtesting_service import BacktestingService
from application.predictive_core import (
    PredictiveCacheState,
    average_correlation,
    compute_ema_prediction,
    extract_backtest_series,
    normalise_symbol_sector,
    run_backtest,
)
from services.cache import CacheService
from services.performance_metrics import track_function
from services.performance_timer import performance_timer
from shared.settings import PREDICTIVE_TTL_HOURS

from predictive_engine import __version__ as ENGINE_VERSION
from predictive_engine.adapters import build_sector_prediction_frame


LOGGER = logging.getLogger(__name__)

_CACHE_NAMESPACE = "predictive"
_CACHE_KEY = "sector_predictions"
_CACHE = CacheService(
    namespace=_CACHE_NAMESPACE,
    ttl_override=PREDICTIVE_TTL_HOURS * 3600.0,
)
_CACHE_STATE = PredictiveCacheState()


def _cache_last_updated(cache: CacheService) -> str | None:
    try:
        value = getattr(cache, "last_updated_human", "-")
    except Exception:  # pragma: no cover - defensive
        return None
    if not value or value == "-":
        return None
    return str(value)


@dataclass(frozen=True)
class PredictiveSnapshot:
    """Container that surfaces statistics for cached predictions."""

    hits: int
    misses: int
    last_updated: str = "-"
    ttl_hours: float | None = None
    remaining_ttl: float | None = None

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_ratio(self) -> float:
        """Return the percentage of cache hits over total lookups."""

        total = self.total
        if total <= 0:
            return 0.0
        return float(self.hits) / float(total)

    def as_dict(self) -> dict[str, float | int | str]:
        """Expose snapshot information as a serialisable mapping."""

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hit_ratio,
            "last_updated": self.last_updated,
            "ttl_hours": self.ttl_hours,
            "remaining_ttl": self.remaining_ttl,
        }


def _prepare_opportunities(opportunities: pd.DataFrame | None) -> pd.DataFrame:
    frame = normalise_symbol_sector(opportunities)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=["symbol", "sector"])
    required = [column for column in ["symbol", "sector"] if column in frame.columns]
    if len(required) < 2:
        return pd.DataFrame(columns=["symbol", "sector"])
    return frame.loc[:, required].copy()


@track_function("predict_sector_performance")
def predict_sector_performance(
    opportunities: pd.DataFrame | None,
    *,
    backtesting_service: BacktestingService | None = None,
    cache: CacheService | None = None,
    span: int = 10,
    ttl_hours: float | None = None,
) -> pd.DataFrame:
    """Return expected sector returns using EMA-smoothed backtests."""

    effective_ttl_hours = (
        float(ttl_hours)
        if ttl_hours is not None
        else float(PREDICTIVE_TTL_HOURS)
    )
    ttl_seconds = max(effective_ttl_hours, 0.0) * 3600.0
    active_cache = cache or _CACHE
    if isinstance(active_cache, CacheService):
        active_cache.set_ttl_override(ttl_seconds)
    telemetry: dict[str, object] = {
        "status": "success",
        "span": int(span),
        "ttl_hours": float(effective_ttl_hours),
    }
    with performance_timer("predictive_compute", extra=telemetry):
        cached = active_cache.get(_CACHE_KEY)
        if isinstance(cached, pd.DataFrame):
            telemetry["cache"] = "hit"
            _CACHE_STATE.record_hit(
                last_updated=_cache_last_updated(active_cache),
                ttl_hours=effective_ttl_hours,
            )
            return cached.copy()

        telemetry["cache"] = "miss"
        frame = _prepare_opportunities(opportunities)
        telemetry["opportunities"] = int(len(frame))
        if frame.empty:
            empty = pd.DataFrame(
                columns=["sector", "predicted_return", "sample_size", "avg_correlation", "confidence"],
            )
            active_cache.set(_CACHE_KEY, empty, ttl=ttl_seconds)
            _CACHE_STATE.record_miss(
                last_updated=_cache_last_updated(active_cache),
                ttl_hours=effective_ttl_hours,
            )
            telemetry["result_rows"] = 0
            return empty

        LOGGER.debug("Ejecutando predictive engine %s", ENGINE_VERSION)
        service = backtesting_service or BacktestingService()
        predictions = build_sector_prediction_frame(
            frame,
            backtesting_service=service,
            run_backtest=lambda svc, symbol: run_backtest(svc, symbol, logger=LOGGER),
            extract_series=lambda backtest, column: extract_backtest_series(backtest, column),
            ema_predictor=lambda series, ema_span: compute_ema_prediction(series, span=ema_span),
            average_correlation=average_correlation,
            span=span,
        )

        telemetry["result_rows"] = int(len(predictions))
        active_cache.set(_CACHE_KEY, predictions, ttl=ttl_seconds)
        _CACHE_STATE.record_miss(
            last_updated=_cache_last_updated(active_cache),
            ttl_hours=effective_ttl_hours,
        )
        return predictions.copy()


def get_cache_stats() -> PredictiveSnapshot:
    """Expose current cache counters for predictive workloads."""

    last_updated = _CACHE_STATE.last_updated
    if not last_updated or last_updated == "-":
        cached_value = _cache_last_updated(_CACHE)
        if cached_value is not None:
            last_updated = cached_value
        else:
            last_updated = "-"
    ttl_hours = _CACHE_STATE.ttl_hours
    if ttl_hours is None:
        try:
            effective_ttl = _CACHE.get_effective_ttl()
        except Exception:  # pragma: no cover - defensive
            effective_ttl = None
        if effective_ttl is not None:
            ttl_hours = float(effective_ttl) / 3600.0
    remaining_ttl: float | None = None
    try:
        remaining_ttl = _CACHE.remaining_ttl()
    except Exception:  # pragma: no cover - defensive safeguard
        remaining_ttl = None
    return PredictiveSnapshot(
        hits=_CACHE_STATE.hits,
        misses=_CACHE_STATE.misses,
        last_updated=last_updated,
        ttl_hours=ttl_hours,
        remaining_ttl=remaining_ttl,
    )


def reset_cache() -> None:
    """Utility mainly intended for tests to clear predictive caches."""

    global _CACHE_STATE
    _CACHE.clear()
    _CACHE.set_ttl_override(PREDICTIVE_TTL_HOURS * 3600.0)
    _CACHE_STATE = PredictiveCacheState()


__all__ = [
    "PredictiveSnapshot",
    "predict_sector_performance",
    "get_cache_stats",
    "reset_cache",
]
