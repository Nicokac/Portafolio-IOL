"""Sector-level predictive analytics with adaptive caching."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
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
from shared.settings import PREDICTIVE_TTL_HOURS


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


def _aggregate_sector_predictions(
    frame: pd.DataFrame,
    *,
    backtesting_service: BacktestingService,
    span: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for sector, group in frame.groupby("sector"):
        symbol_returns: dict[str, pd.Series] = {}
        symbol_predictions: list[tuple[str, float]] = []

        for symbol in group["symbol"]:
            symbol_str = str(symbol)
            backtest = run_backtest(
                backtesting_service,
                symbol_str,
                logger=LOGGER,
            )
            if backtest is None:
                continue
            returns = extract_backtest_series(backtest, "strategy_ret")
            if returns.empty:
                continue
            predicted = compute_ema_prediction(returns, span=span)
            if predicted is None:
                continue
            symbol_returns[symbol_str] = returns
            symbol_predictions.append((symbol_str, predicted))

        if not symbol_predictions:
            continue

        avg_corr = average_correlation(symbol_returns)
        if avg_corr.empty:
            avg_corr = pd.Series(
                0.0,
                index=[symbol for symbol, _ in symbol_predictions],
                dtype=float,
            )

        weights: list[float] = []
        predictions: list[float] = []
        for symbol, predicted in symbol_predictions:
            correlation = float(avg_corr.get(symbol, 0.0)) if isinstance(avg_corr, pd.Series) else 0.0
            penalty = max(correlation, 0.0)
            weight = 1.0 / (1.0 + penalty)
            weights.append(weight)
            predictions.append(predicted)

        weights_array = np.array(weights, dtype=float)
        if not np.isfinite(weights_array).all() or weights_array.sum() <= 0:
            weights_array = np.ones_like(weights_array)
        weights_array = weights_array / weights_array.sum()

        predicted_sector = float(np.dot(weights_array, predictions))
        avg_corr_value = float(np.nanmean(avg_corr.to_numpy())) if not avg_corr.empty else 0.0
        confidence = float(max(0.0, min(1.0, 1.0 - max(avg_corr_value, 0.0))))

        rows.append(
            {
                "sector": str(sector),
                "predicted_return": predicted_sector,
                "sample_size": len(symbol_predictions),
                "avg_correlation": avg_corr_value,
                "confidence": confidence,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["sector", "predicted_return", "sample_size", "avg_correlation", "confidence"],
        )

    result = pd.DataFrame(rows)
    result = result.sort_values("predicted_return", ascending=False).reset_index(drop=True)
    return result


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
    cached = active_cache.get(_CACHE_KEY)
    if isinstance(cached, pd.DataFrame):
        _CACHE_STATE.record_hit(
            last_updated=_cache_last_updated(active_cache),
            ttl_hours=effective_ttl_hours,
        )
        return cached.copy()

    frame = _prepare_opportunities(opportunities)
    if frame.empty:
        empty = pd.DataFrame(
            columns=["sector", "predicted_return", "sample_size", "avg_correlation", "confidence"],
        )
        active_cache.set(_CACHE_KEY, empty, ttl=ttl_seconds)
        _CACHE_STATE.record_miss(
            last_updated=_cache_last_updated(active_cache),
            ttl_hours=effective_ttl_hours,
        )
        return empty

    service = backtesting_service or BacktestingService()
    predictions = _aggregate_sector_predictions(
        frame,
        backtesting_service=service,
        span=span,
    )

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
