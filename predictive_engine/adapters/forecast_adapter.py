"""Adapters for the adaptive forecasting engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Tuple

import pandas as pd

from predictive_engine.base import calculate_adaptive_forecast, update_adaptive_state
from predictive_engine.models import (
    AdaptiveForecastResult,
    AdaptiveState,
    AdaptiveUpdateResult,
    empty_history_frame,
)
from predictive_engine.storage import load_forecast_history, save_forecast_history
from services.performance_metrics import measure_execution
from services.performance_timer import profile_block


@dataclass
class EngineUpdateContext:
    """Context information required to interact with cache services."""

    cache: Any | None
    state_key: str
    correlation_key: str
    ttl_seconds: float
    max_history_rows: int
    persist: bool = True
    persist_history: bool = False
    history_path: str | Path | None = None
    history_loader: Callable[[str | Path], pd.DataFrame] | None = None
    history_saver: Callable[[pd.DataFrame, str | Path], object] | None = None
    warm_start: bool = False
    state: AdaptiveState | None = field(default=None, init=False)
    _initialised_from_storage: bool = field(default=False, init=False, repr=False)

    def resolve_state(self) -> Tuple[AdaptiveState, bool]:
        if self.state is not None:
            return self.state, True
        cached_state = None
        if self.cache is not None:
            try:
                cached_state = self.cache.get(self.state_key)
            except Exception:
                cached_state = None
        if cached_state is not None:
            try:
                self.state = _coerce_state(cached_state)
            except Exception:
                self.state = None
            else:
                return self.state, True
        if (
            not self._initialised_from_storage
            and self.warm_start
            and self.history_loader is not None
            and self.history_path
        ):
            try:
                persisted = self.history_loader(self.history_path)
            except Exception:
                persisted = empty_history_frame()
            persisted = persisted.copy()
            if not persisted.empty:
                persisted = persisted.sort_values("timestamp").tail(
                    self.max_history_rows
                )
                timestamps = pd.to_datetime(
                    persisted.get("timestamp"),
                    errors="coerce",
                )
                if timestamps.dropna().empty:
                    last_timestamp = None
                else:
                    last_timestamp = timestamps.dropna().iloc[-1]
                self.state = AdaptiveState(
                    history=persisted,
                    last_updated=last_timestamp,
                )
                self._initialised_from_storage = True
                return self.state, False
            self._initialised_from_storage = True
        self.state = AdaptiveState(history=empty_history_frame(), last_updated=None)
        return self.state, False

    def persist_state(self, result: AdaptiveUpdateResult) -> None:
        self.state = result.state.copy()
        if self.persist_history and self.history_saver and self.history_path:
            try:
                self.history_saver(self.state.history.copy(), self.history_path)
            except Exception:
                pass
        if not self.persist or self.cache is None:
            return
        try:
            self.cache.set(self.state_key, result.state, ttl=self.ttl_seconds)
            self.cache.set(
                self.correlation_key,
                result.correlation_matrix.copy(),
                ttl=self.ttl_seconds,
            )
        except Exception:
            # Cache persistence is best-effort.
            return


def update_model_with_cache(
    predictions: pd.DataFrame | None,
    actuals: pd.DataFrame | None,
    *,
    context: EngineUpdateContext,
    ema_span: int,
    timestamp: pd.Timestamp | None,
    performance_prefix: str = "adaptive",
) -> Tuple[AdaptiveUpdateResult, bool]:
    fetch_label = f"{performance_prefix}.fetch"
    model_label = f"{performance_prefix}.modeling"
    persist_label = f"{performance_prefix}.persistence"
    with profile_block(
        fetch_label,
        extra={"stage": "fetch"},
        module=__name__,
    ):
        state, cache_hit = context.resolve_state()
    with profile_block(
        model_label,
        extra={"stage": "modeling"},
        module=__name__,
    ):
        result = update_adaptive_state(
            predictions,
            actuals,
            state=state,
            ema_span=max(int(ema_span), 1),
            timestamp=timestamp,
            max_history_rows=context.max_history_rows,
        )
    with profile_block(
        persist_label,
        extra={"stage": "persistence", "cache_hit": str(bool(cache_hit))},
        module=__name__,
    ):
        context.persist_state(result)
    return result, cache_hit


def build_adaptive_updater(
    *,
    context: EngineUpdateContext,
    ema_span: int,
    performance_prefix: str = "adaptive",
) -> Callable[[pd.DataFrame, pd.DataFrame, pd.Timestamp], AdaptiveUpdateResult]:
    def _updater(
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> AdaptiveUpdateResult:
        result, _ = update_model_with_cache(
            predictions,
            actuals,
            context=context,
            ema_span=ema_span,
            timestamp=timestamp,
            performance_prefix=performance_prefix,
        )
        return result

    return _updater


def _coerce_state(value: object) -> AdaptiveState:
    if isinstance(value, AdaptiveState):
        return value.copy()
    history = empty_history_frame()
    last_updated = getattr(value, "last_updated", None)
    maybe_history = getattr(value, "history", None)
    if isinstance(value, Mapping):
        maybe_history = value.get("history", maybe_history)
        last_updated = value.get("last_updated", last_updated)
    if isinstance(maybe_history, pd.DataFrame):
        history = maybe_history.copy()
    return AdaptiveState(history=history, last_updated=last_updated)


def _extract_cache_metadata(cache: Any | None) -> Dict[str, float | str]:
    hit_ratio = 0.0
    last_updated = "-"
    if cache is None:
        return {"hit_ratio": hit_ratio, "last_updated": last_updated}
    if hasattr(cache, "hit_ratio"):
        try:
            hit_ratio = float(cache.hit_ratio())
        except Exception:
            hit_ratio = 0.0
    if hasattr(cache, "last_updated_human"):
        try:
            last_updated = str(cache.last_updated_human)
        except Exception:
            last_updated = "-"
    return {"hit_ratio": hit_ratio, "last_updated": last_updated}


def run_adaptive_forecast(
    *,
    history: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    actuals: pd.DataFrame | None = None,
    cache: Any | None = None,
    state_key: str = "adaptive_state",
    correlation_key: str = "adaptive_correlations",
    ema_span: int = 5,
    rolling_window: int = 20,
    ttl_hours: float | None = None,
    max_history_rows: int = 720,
    persist_state: bool = True,
    persist_history: bool = False,
    history_path: str | Path | None = "./data/forecast_history.parquet",
    warm_start: bool = True,
    timestamp: pd.Timestamp | None = None,
    performance_prefix: str = "adaptive",
) -> Dict[str, Any]:
    """Execute adaptive update and/or forecast with persistence helpers."""

    has_history = isinstance(history, pd.DataFrame) and not history.empty
    has_inputs = (
        isinstance(predictions, pd.DataFrame)
        or isinstance(actuals, pd.DataFrame)
    )
    if has_history and has_inputs:
        raise ValueError(
            "history and (predictions, actuals) are mutually exclusive inputs"
        )

    ttl_seconds = max(float(ttl_hours) if ttl_hours is not None else 0.0, 0.0) * 3600.0
    if cache is not None and hasattr(cache, "set_ttl_override"):
        try:
            cache.set_ttl_override(ttl_seconds)
        except Exception:
            pass

    context = EngineUpdateContext(
        cache=cache,
        state_key=state_key,
        correlation_key=correlation_key,
        ttl_seconds=ttl_seconds,
        max_history_rows=max_history_rows,
        persist=persist_state,
        persist_history=persist_history,
        history_path=history_path,
        history_loader=load_forecast_history if warm_start else None,
        history_saver=save_forecast_history if persist_history else None,
        warm_start=warm_start,
    )

    cache_metadata = _extract_cache_metadata(cache)

    if has_inputs:
        with measure_execution(f"{performance_prefix}.update"):
            result, cache_hit = update_model_with_cache(
                predictions,
                actuals,
                context=context,
                ema_span=ema_span,
                timestamp=timestamp,
                performance_prefix=performance_prefix,
            )
        return {
            "update": result,
            "cache_hit": cache_hit,
            "cache_metadata": cache_metadata,
        }

    cache_hit_flag = False

    def _updater(
        preds: pd.DataFrame,
        acts: pd.DataFrame,
        ts: pd.Timestamp,
    ) -> AdaptiveUpdateResult:
        nonlocal cache_hit_flag
        result, hit = update_model_with_cache(
            preds,
            acts,
            context=context,
            ema_span=ema_span,
            timestamp=ts,
            performance_prefix=performance_prefix,
        )
        cache_hit_flag = cache_hit_flag or hit
        return result

    with profile_block(
        f"{performance_prefix}.calculation",
        extra={"stage": "calculation"},
        module=__name__,
    ):
        forecast_result: AdaptiveForecastResult = calculate_adaptive_forecast(
            history if has_history else pd.DataFrame(),
            ema_span=ema_span,
            rolling_window=rolling_window,
            model_updater=_updater,
            cache_metadata=cache_metadata,
            persist_history=persist_history,
            history_path=history_path,
            history_saver=context.history_saver,
        )

    return {
        "forecast": forecast_result,
        "cache_hit": cache_hit_flag,
        "cache_metadata": cache_metadata,
    }
