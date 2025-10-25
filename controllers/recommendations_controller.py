"""Controller helpers exposing recommendation view-models for the UI."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from application.adaptive_predictive_service import simulate_adaptive_forecast
from application.predictive_service import (
    PredictiveCacheSnapshot,
    PredictiveSnapshot,
    build_adaptive_history,
    get_cache_stats,
    predict_sector_performance,
    predictive_job_status,
)
from ui.utils.formatters import normalise_hit_ratio, resolve_badge_state

LOGGER = logging.getLogger(__name__)


@dataclass
class PredictiveCacheViewModel:
    """Serializable representation of the predictive cache status."""

    hits: int = 0
    misses: int = 0
    last_updated: str = "-"
    ttl_hours: float | None = None
    remaining_ttl: float | None = None

    @property
    def hit_ratio(self) -> float:
        total = int(self.hits) + int(self.misses)
        if total <= 0:
            return 0.0
        return float(self.hits) / float(total)

    def to_dict(self) -> dict[str, Any]:
        ratio = normalise_hit_ratio(self.hit_ratio)
        return {
            "hits": int(self.hits),
            "misses": int(self.misses),
            "last_updated": self.last_updated,
            "ttl_hours": self.ttl_hours,
            "remaining_ttl": self.remaining_ttl,
            "hit_ratio": ratio,
            "badge_state": resolve_badge_state(ratio),
        }


@dataclass
class SectorPerformanceViewModel:
    """View payload combining sector predictions and cache metadata."""

    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    cache: PredictiveCacheViewModel = field(default_factory=PredictiveCacheViewModel)
    job_status: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions": self.predictions.copy(),
            "cache": self.cache.to_dict(),
            "job_status": dict(self.job_status),
        }


@dataclass
class AdaptiveForecastViewModel:
    """Normalised payload for the adaptive forecast panel."""

    payload: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    cache_metadata: dict[str, Any] = field(default_factory=dict)


def _snapshot_to_view(
    snapshot: PredictiveCacheSnapshot | PredictiveSnapshot | Mapping[str, Any] | None,
) -> PredictiveCacheViewModel:
    if snapshot is None:
        return PredictiveCacheViewModel()
    if isinstance(snapshot, PredictiveCacheSnapshot):
        data = snapshot.to_dict()
    elif isinstance(snapshot, PredictiveSnapshot):
        data = snapshot.as_dict()
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


def build_adaptive_history_view(
    opportunities: pd.DataFrame | None,
    recommendations: pd.DataFrame | None,
    *,
    profile: Mapping[str, Any] | None = None,
    span: int = 5,
    max_symbols: int = 12,
    periods: int = 6,
) -> tuple[pd.DataFrame, bool]:
    """Return adaptive history frame and whether it was synthetically generated."""

    context: dict[str, Any] = {}
    if isinstance(profile, Mapping):
        context["profile"] = dict(profile)

    history = build_adaptive_history(
        opportunities,
        mode="real",
        span=span,
        max_symbols=max_symbols,
        context=context or None,
    )
    if not history.empty:
        return history, False

    symbols: list[str] = []
    sectors: list[str] = []
    if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
        if "symbol" in recommendations.columns:
            symbols = (
                recommendations.get("symbol", pd.Series(dtype=str))
                .astype("string")
                .dropna()
                .astype(str)
                .str.upper()
                .str.strip()
                .unique()
                .tolist()
            )
        if "sector" in recommendations.columns:
            sectors = (
                recommendations.get("sector", pd.Series(dtype=str))
                .astype("string")
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

    LOGGER.warning(
        "Histórico adaptativo real vacío, generando histórico sintético",
        extra={
            "symbols": symbols[:30],
            "sectors": sectors[:30],
            "profile": profile,
        },
    )

    synthetic = build_adaptive_history(
        recommendations,
        mode="synthetic",
        periods=periods,
        context=context or None,
    )
    return synthetic, True


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
        background=True,
    )
    job_status: dict[str, Any] = {}
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions)
    else:
        metadata = predictions.attrs.get("predictive_job")
        predictions = predictions.copy()
        if isinstance(metadata, Mapping):
            job_status = dict(metadata)
            snapshot = predictive_job_status(job_status.get("job_id"))
            if isinstance(snapshot, Mapping):
                for key, value in snapshot.items():
                    job_status.setdefault(key, value)
    cache_view = _snapshot_to_view(get_cache_stats())
    return SectorPerformanceViewModel(
        predictions=predictions,
        cache=cache_view,
        job_status=job_status,
    )


def resolve_predictive_spinner(job_status: Mapping[str, Any] | None) -> str | None:
    """Return a spinner message when background predictive jobs are active."""

    if not isinstance(job_status, Mapping) or not job_status:
        return None
    status = str(job_status.get("status") or "").lower()
    if status in {"pending", "running"}:
        job_reference = job_status.get("job_id") or job_status.get("latest_job_id")
        if job_reference:
            return f"Calculando predicciones en segundo plano (job {job_reference})..."
        return "Calculando predicciones en segundo plano..."
    if status in {"failed", "cancelled"}:
        return "No se pudieron actualizar las predicciones sectoriales."
    return None


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
    context: Mapping[str, Any] | None = None,
) -> AdaptiveForecastViewModel:
    try:
        payload = simulate_adaptive_forecast(
            history,
            ema_span=ema_span,
            cache=cache,
            persist=persist,
            rolling_window=rolling_window,
            ttl_hours=ttl_hours,
        )
    except Exception:
        log_context: dict[str, Any] = {
            "ema_span": ema_span,
            "persist": persist,
            "rolling_window": rolling_window,
        }
        if isinstance(context, Mapping):
            for key in ("symbols", "sectors", "profile"):
                if key in context:
                    log_context[key] = context[key]
        LOGGER.error(
            "Error al simular el pronóstico adaptativo",
            extra=log_context,
            exc_info=True,
        )
        raise
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
    "build_adaptive_history_view",
    "load_sector_performance_view",
    "get_predictive_cache_view",
    "run_adaptive_forecast_view",
    "resolve_predictive_spinner",
]
