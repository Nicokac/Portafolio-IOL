"""Lightweight QA profiling hooks for baseline performance metrics."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Dict, Iterator

from shared.telemetry import log as telemetry_log

logger = logging.getLogger(__name__)

_QA_EVENT_NAME = "qa_profiler"
_METRIC_FIELDS = (
    "startup_time_ms",
    "ui_render_time_ms",
    "cache_load_time_ms",
    "auth_latency_ms",
)

_startup_started_at = time.perf_counter()
_metrics: Dict[str, float] = {}


def _snapshot(event: str) -> None:
    """Emit the current metric snapshot through the telemetry channel."""

    payload = {name: _metrics.get(name) for name in _METRIC_FIELDS}
    telemetry_log(event, qa=True, **payload)


def _record_metric(name: str, duration_ms: float) -> None:
    if name in _metrics:
        return
    coerced = max(float(duration_ms), 0.0)
    _metrics[name] = coerced
    logger.debug("QA metric recorded %s=%.3f", name, coerced)


def record_startup_complete() -> None:
    """Capture the elapsed startup time once the bootstrap finishes."""

    elapsed_ms = (time.perf_counter() - _startup_started_at) * 1000.0
    _record_metric("startup_time_ms", elapsed_ms)
    _snapshot(f"{_QA_EVENT_NAME}_startup")


@contextmanager
def track_ui_render() -> Iterator[None]:
    """Profile the first UI render cycle."""

    if "ui_render_time_ms" in _metrics:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _record_metric("ui_render_time_ms", elapsed_ms)
        _snapshot(f"{_QA_EVENT_NAME}_ui")


@contextmanager
def track_cache_load() -> Iterator[None]:
    """Profile the latency of the first cache hydration."""

    if "cache_load_time_ms" in _metrics:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _record_metric("cache_load_time_ms", elapsed_ms)
        _snapshot(f"{_QA_EVENT_NAME}_cache")


@contextmanager
def track_auth_latency() -> Iterator[None]:
    """Profile the time spent performing authentication."""

    if "auth_latency_ms" in _metrics:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _record_metric("auth_latency_ms", elapsed_ms)
        _snapshot(f"{_QA_EVENT_NAME}_auth")


def _reset_for_tests() -> None:
    """Test helper to reset profiler state."""

    global _startup_started_at
    _startup_started_at = time.perf_counter()
    _metrics.clear()


__all__ = [
    "record_startup_complete",
    "track_ui_render",
    "track_cache_load",
    "track_auth_latency",
]
