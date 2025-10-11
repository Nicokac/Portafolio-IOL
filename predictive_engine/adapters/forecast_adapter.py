"""Adapters for the adaptive forecasting engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Tuple

import pandas as pd

from predictive_engine.base import update_adaptive_state
from predictive_engine.models import AdaptiveState, AdaptiveUpdateResult, empty_history_frame


@dataclass
class EngineUpdateContext:
    """Context information required to interact with cache services."""

    cache: Any | None
    state_key: str
    correlation_key: str
    ttl_seconds: float
    max_history_rows: int
    persist: bool = True
    state: AdaptiveState | None = field(default=None, init=False)

    def resolve_state(self) -> Tuple[AdaptiveState, bool]:
        if self.state is not None:
            return self.state, True
        cached_state = None
        cache_hit = False
        if self.cache is not None:
            try:
                cached_state = self.cache.get(self.state_key)
            except Exception:
                cached_state = None
            cache_hit = isinstance(cached_state, AdaptiveState)
        if cache_hit:
            self.state = cached_state.copy()  # type: ignore[union-attr]
            return self.state, True
        self.state = AdaptiveState(history=empty_history_frame(), last_updated=None)
        return self.state, False

    def persist_state(self, result: AdaptiveUpdateResult) -> None:
        self.state = result.state.copy()
        if not self.persist or self.cache is None:
            return
        try:
            self.cache.set(self.state_key, result.state, ttl=self.ttl_seconds)
            self.cache.set(self.correlation_key, result.correlation_matrix.copy(), ttl=self.ttl_seconds)
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
) -> Tuple[AdaptiveUpdateResult, bool]:
    state, cache_hit = context.resolve_state()
    result = update_adaptive_state(
        predictions,
        actuals,
        state=state,
        ema_span=max(int(ema_span), 1),
        timestamp=timestamp,
        max_history_rows=context.max_history_rows,
    )
    context.persist_state(result)
    return result, cache_hit


def build_adaptive_updater(
    *,
    context: EngineUpdateContext,
    ema_span: int,
) -> Callable[[pd.DataFrame, pd.DataFrame, pd.Timestamp], AdaptiveUpdateResult]:
    def _updater(predictions: pd.DataFrame, actuals: pd.DataFrame, timestamp: pd.Timestamp) -> AdaptiveUpdateResult:
        result, _ = update_model_with_cache(
            predictions,
            actuals,
            context=context,
            ema_span=ema_span,
            timestamp=timestamp,
        )
        return result

    return _updater
