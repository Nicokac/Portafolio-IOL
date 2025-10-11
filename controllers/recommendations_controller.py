"""Controller helpers exposing recommendation view-models for the UI."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping

import pandas as pd

from application.adaptive_predictive_service import simulate_adaptive_forecast
from application.predictive_service import (
    PredictiveSnapshot,
    get_cache_stats,
    predict_sector_performance,
)


@dataclass
class PredictiveCacheViewModel:
    """Serializable representation of the predictive cache status."""

    hits: int = 0
    misses: int = 0
    last_updated: str = "-"
    ttl_hours: float | None = None
    remaining_ttl: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": int(self.hits),
            "misses": int(self.misses),
            "last_updated": self.last_updated,
            "ttl_hours": self.ttl_hours,
            "remaining_ttl": self.remaining_ttl,
        }


@dataclass
class SectorPerformanceViewModel:
    """View payload combining sector predictions and cache metadata."""

    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    cache: PredictiveCacheViewModel = field(default_factory=PredictiveCacheViewModel)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions": self.predictions.copy(),
            "cache": self.cache.to_dict(),
        }


@dataclass
class AdaptiveForecastViewModel:
    """Normalised payload for the adaptive forecast panel."""

    payload: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    cache_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload": dict(self.payload),
            "summary": dict(self.summary),
            "cache_metadata": dict(self.cache_metadata),
        }


def _snapshot_to_view(snapshot: PredictiveSnapshot | Mapping[str, Any] | None) -> PredictiveCacheViewModel:
    if snapshot is None:
        return PredictiveCacheViewModel()
    if isinstance(snapshot, PredictiveSnapshot):
        data = asdict(snapshot)
    elif isinstance(snapshot, Mapping):
        data = dict(snapshot)
    else:
        return PredictiveCacheViewModel()
    last_updated = str(data.get("last_updated", "-")) or "-"
    return PredictiveCacheViewModel(
        hits=int(data.get("hits", 0) or 0),
        misses=int(data.get("misses", 0) or 0),
        last_updated=last_updated,
        ttl_hours=data.get("ttl_hours"),
        remaining_ttl=data.get("remaining_ttl"),
    )


def _clone_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _clone_value(val) for key, val in value.items()}
    return value


def load_sector_performance_view(
    opportunities: pd.DataFrame | None,
    *,
    span: int = 10,
    ttl_hours: float | None = None,
) -> SectorPerformanceViewModel:
    predictions = predict_sector_performance(
        opportunities,
        span=span,
        ttl_hours=ttl_hours,
    )
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions)
    else:
        predictions = predictions.copy()
    cache_view = _snapshot_to_view(get_cache_stats())
    return SectorPerformanceViewModel(predictions=predictions, cache=cache_view)


def get_predictive_cache_view() -> PredictiveCacheViewModel:
    snapshot = get_cache_stats()
    return _snapshot_to_view(snapshot)


def run_adaptive_forecast_view(
    history: pd.DataFrame | None,
    *,
    ema_span: int = 4,
    cache: Any | None = None,
    persist: bool = True,
    rolling_window: int = 20,
    ttl_hours: float | None = None,
) -> AdaptiveForecastViewModel:
    payload = simulate_adaptive_forecast(
        history,
        ema_span=ema_span,
        cache=cache,
        persist=persist,
        rolling_window=rolling_window,
        ttl_hours=ttl_hours,
    )
    if not isinstance(payload, Mapping):
        payload = {}
    cloned_payload = {key: _clone_value(value) for key, value in payload.items()}
    summary = cloned_payload.get("summary")
    if not isinstance(summary, Mapping):
        summary = {}
    else:
        summary = dict(summary)
        cloned_payload["summary"] = summary
    cache_metadata = cloned_payload.get("cache_metadata")
    if not isinstance(cache_metadata, Mapping):
        cache_metadata = {}
    else:
        cache_metadata = dict(cache_metadata)
        cloned_payload["cache_metadata"] = cache_metadata
    return AdaptiveForecastViewModel(
        payload=cloned_payload,
        summary=summary,
        cache_metadata=cache_metadata,
    )


__all__ = [
    "PredictiveCacheViewModel",
    "SectorPerformanceViewModel",
    "AdaptiveForecastViewModel",
    "load_sector_performance_view",
    "get_predictive_cache_view",
    "run_adaptive_forecast_view",
]
