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
from domain.adaptive_cache_lock import adaptive_cache_lock
from services.cache import CacheService
from services.cache.market_data_cache import get_market_data_cache
from services.performance_metrics import track_function
from services.performance_timer import performance_timer
from shared.settings import PREDICTIVE_TTL_HOURS

from predictive_engine import __version__ as ENGINE_VERSION
from predictive_engine.adapters import build_sector_prediction_frame


LOGGER = logging.getLogger(__name__)

_CACHE_NAMESPACE = "predictive"
_CACHE_KEY_PREFIX = "sector_predictions"
_MARKET_CACHE = get_market_data_cache()
_CACHE = _MARKET_CACHE.prediction_cache
_CACHE.set_ttl_override(PREDICTIVE_TTL_HOURS * 3600.0)
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
    active_cache: CacheService | None = cache if cache is not None else _CACHE
    if isinstance(active_cache, CacheService):
        active_cache.set_ttl_override(ttl_seconds)
    telemetry: dict[str, object] = {
        "status": "success",
        "span": int(span),
        "ttl_hours": float(effective_ttl_hours),
    }
    frame = _prepare_opportunities(opportunities)
    telemetry["opportunities"] = int(len(frame))
    symbols = (
        frame.get("symbol", pd.Series(dtype=str))
        .astype("string")
        .str.upper()
        .str.strip()
        .tolist()
    )
    sectors = (
        frame.get("sector", pd.Series(dtype=str))
        .astype("string")
        .str.strip()
        .tolist()
    )
    cache_key = f"{_CACHE_KEY_PREFIX}:{_MARKET_CACHE.build_prediction_key(symbols, span=span, sectors=sectors)}"

    LOGGER.debug(
        "Solicitando lock adaptativo para predicciones (span=%s, símbolos=%s, sectores=%s)",
        span,
        symbols,
        sectors,
    )

    cached_frame: pd.DataFrame | None = None
    cache_hit = False

    with adaptive_cache_lock:
        LOGGER.debug(
            "Lock adaptativo adquirido en predictive_service (span=%s, símbolos=%s)",
            span,
            symbols,
        )
        if active_cache is not None:
            cached = active_cache.get(cache_key)
        else:
            cached = None
        if isinstance(cached, pd.DataFrame):
            cached_frame = cached
            cache_hit = True
            telemetry["cache"] = "hit"
            last_updated = (
                _cache_last_updated(active_cache)
                if isinstance(active_cache, CacheService)
                else None
            )
            _CACHE_STATE.record_hit(
                last_updated=last_updated,
                ttl_hours=effective_ttl_hours,
            )
        LOGGER.debug(
            "Liberando lock adaptativo en predictive_service tras consulta de caché"
        )

    if cache_hit and isinstance(cached_frame, pd.DataFrame):
        LOGGER.debug(
            "Predicciones recuperadas desde caché (span=%s, símbolos=%s, sectores=%s)",
            span,
            symbols,
            sectors,
        )
        return cached_frame.copy()

    telemetry["cache"] = "miss"

    if frame.empty:
        empty = pd.DataFrame(
            columns=[
                "sector",
                "predicted_return",
                "sample_size",
                "avg_correlation",
                "confidence",
            ]
        )
        with adaptive_cache_lock:
            LOGGER.debug(
                "Lock adaptativo adquirido para persistir predicciones vacías (span=%s)",
                span,
            )
            if active_cache is not None:
                active_cache.set(cache_key, empty, ttl=ttl_seconds)
            last_updated = (
                _cache_last_updated(active_cache)
                if isinstance(active_cache, CacheService)
                else None
            )
            _CACHE_STATE.record_miss(
                last_updated=last_updated,
                ttl_hours=effective_ttl_hours,
            )
            LOGGER.debug(
                "Lock adaptativo liberado tras almacenar predicciones vacías"
            )
        telemetry["result_rows"] = 0
        return empty

    LOGGER.debug(
        "Ejecutando predictive engine %s (span=%s, símbolos=%s, sectores=%s)",
        ENGINE_VERSION,
        span,
        symbols,
        sectors,
    )

    service = backtesting_service or BacktestingService()

    with performance_timer("predictive_compute", extra=telemetry):
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

    stored_frame: pd.DataFrame = predictions
    reused = False

    with adaptive_cache_lock:
        LOGGER.debug(
            "Lock adaptativo adquirido para actualizar caché predictivo (span=%s)",
            span,
        )
        existing = active_cache.get(cache_key) if active_cache is not None else None
        if isinstance(existing, pd.DataFrame):
            stored_frame = existing
            reused = True
            telemetry["cache"] = "hit"
            last_updated = (
                _cache_last_updated(active_cache)
                if isinstance(active_cache, CacheService)
                else None
            )
            _CACHE_STATE.record_hit(
                last_updated=last_updated,
                ttl_hours=effective_ttl_hours,
            )
        else:
            if active_cache is not None:
                active_cache.set(cache_key, predictions, ttl=ttl_seconds)
            last_updated = (
                _cache_last_updated(active_cache)
                if isinstance(active_cache, CacheService)
                else None
            )
            _CACHE_STATE.record_miss(
                last_updated=last_updated,
                ttl_hours=effective_ttl_hours,
            )
        LOGGER.debug(
            "Lock adaptativo liberado tras actualizar caché predictivo"
        )

    if reused:
        LOGGER.debug(
            "Predicciones ya presentes en caché reutilizadas (span=%s, símbolos=%s)",
            span,
            symbols,
        )
    else:
        LOGGER.debug(
            "Predicciones almacenadas en caché (span=%s, símbolos=%s, sectores=%s)",
            span,
            symbols,
            sectors,
        )

    return stored_frame.copy()


def update_cache_metrics(cache: CacheService | None = None) -> PredictiveSnapshot:
    """Refresh predictive cache metrics without recomputing predictions."""

    active_cache: CacheService | None = cache if cache is not None else _CACHE
    with adaptive_cache_lock:
        LOGGER.debug(
            "Lock adaptativo adquirido para refrescar métricas de caché predictivo"
        )
        hits = _CACHE_STATE.hits
        misses = _CACHE_STATE.misses
        last_updated = _CACHE_STATE.last_updated or "-"
        ttl_hours = _CACHE_STATE.ttl_hours
        remaining_ttl: float | None = None

        if isinstance(active_cache, CacheService):
            try:
                hits = int(getattr(active_cache, "hits", hits))
            except Exception:  # pragma: no cover - defensive
                hits = _CACHE_STATE.hits
            try:
                misses = int(getattr(active_cache, "misses", misses))
            except Exception:  # pragma: no cover - defensive
                misses = _CACHE_STATE.misses
            cache_last_updated = _cache_last_updated(active_cache)
            if cache_last_updated:
                last_updated = cache_last_updated
            try:
                effective_ttl = active_cache.get_effective_ttl()
            except Exception:  # pragma: no cover - defensive
                effective_ttl = None
            if effective_ttl is not None:
                ttl_hours = float(effective_ttl) / 3600.0
            try:
                remaining_ttl = active_cache.remaining_ttl()
            except Exception:  # pragma: no cover - defensive safeguard
                remaining_ttl = None

        _CACHE_STATE.hits = hits
        _CACHE_STATE.misses = misses
        _CACHE_STATE.last_updated = last_updated or "-"
        _CACHE_STATE.ttl_hours = ttl_hours

        snapshot = PredictiveSnapshot(
            hits=hits,
            misses=misses,
            last_updated=_CACHE_STATE.last_updated,
            ttl_hours=_CACHE_STATE.ttl_hours,
            remaining_ttl=remaining_ttl,
        )

        LOGGER.debug(
            "Lock adaptativo liberado tras refrescar métricas de caché predictivo"
        )
    return snapshot


def get_cache_stats() -> PredictiveSnapshot:
    """Expose current cache counters for predictive workloads."""

    with adaptive_cache_lock:
        last_updated = _CACHE_STATE.last_updated
        if not last_updated or last_updated == "-":
            cache_last = _cache_last_updated(_CACHE)
            if cache_last:
                last_updated = cache_last
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

        snapshot = PredictiveSnapshot(
            hits=_CACHE_STATE.hits,
            misses=_CACHE_STATE.misses,
            last_updated=last_updated,
            ttl_hours=ttl_hours,
            remaining_ttl=remaining_ttl,
        )
    return snapshot


def reset_cache() -> None:
    """Utility mainly intended for tests to clear predictive caches."""

    global _CACHE_STATE
    with adaptive_cache_lock:
        LOGGER.debug("Lock adaptativo adquirido para reiniciar el caché predictivo")
        _CACHE.clear()
        _CACHE.set_ttl_override(PREDICTIVE_TTL_HOURS * 3600.0)
        _CACHE_STATE = PredictiveCacheState()
        LOGGER.debug("Lock adaptativo liberado tras reiniciar el caché predictivo")


__all__ = [
    "PredictiveSnapshot",
    "predict_sector_performance",
    "update_cache_metrics",
    "get_cache_stats",
    "reset_cache",
]
