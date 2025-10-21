"""Lightweight QA profiling hooks for baseline performance metrics."""

from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator

try:  # pragma: no cover - optional dependency in some environments
    import psutil  # type: ignore
except Exception:  # pragma: no cover - graceful fallback when psutil unavailable
    psutil = None  # type: ignore[assignment]

try:  # pragma: no cover - ``resource`` not always available (e.g. Windows)
    import resource  # type: ignore
except Exception:  # pragma: no cover - graceful fallback
    resource = None  # type: ignore[assignment]

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
_peak_ram_mb: float = 0.0


def _snapshot(event: str) -> None:
    """Emit the current metric snapshot through the telemetry channel."""

    payload = {name: _metrics.get(name) for name in _METRIC_FIELDS}
    payload.update(_collect_runtime_stats())
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
    global _peak_ram_mb
    _peak_ram_mb = 0.0


def _collect_runtime_stats() -> Dict[str, float]:
    """Return runtime diagnostics captured during each snapshot."""

    stats: Dict[str, float] = {}

    current_ram = _measure_memory_usage_mb()
    global _peak_ram_mb
    if current_ram is not None:
        _peak_ram_mb = max(_peak_ram_mb, current_ram)
    stats["peak_ram_mb"] = _peak_ram_mb if _peak_ram_mb > 0.0 else 0.0

    try:
        stats["active_threads"] = float(threading.active_count())
    except Exception:  # pragma: no cover - defensive safeguard
        stats["active_threads"] = 0.0

    stats["cached_objects"] = float(_resolve_cached_objects())
    return stats


def _measure_memory_usage_mb() -> float | None:
    """Return the current RSS usage in megabytes when available."""

    if psutil is not None:  # pragma: no branch - simple fast path
        try:
            process = psutil.Process()
            rss_bytes = float(process.memory_info().rss)
            if rss_bytes > 0.0:
                return rss_bytes / (1024.0 ** 2)
        except Exception:  # pragma: no cover - psutil runtime failure safeguard
            logger.debug("Unable to determine memory usage via psutil", exc_info=True)

    if resource is not None:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]
            rss_value = float(getattr(usage, "ru_maxrss", 0.0))
            if rss_value <= 0.0:
                return None
            if sys.platform == "darwin":
                return rss_value / (1024.0 ** 2)
            return rss_value / 1024.0
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug("Unable to determine memory usage via resource", exc_info=True)

    return None


def _resolve_cached_objects() -> int:
    """Return the number of cached visual entries available in the session."""

    try:
        from shared.cache import visual_cache_registry  # type: ignore
    except Exception:  # pragma: no cover - cache layer optional during tests
        return 0

    try:
        snapshot = visual_cache_registry.snapshot()
    except Exception:  # pragma: no cover - registry may fail if session not initialised
        logger.debug("Unable to obtain visual cache snapshot", exc_info=True)
        return 0

    entries = snapshot.get("entries")
    if isinstance(entries, dict):
        return sum(1 for value in entries.values() if isinstance(value, dict))
    return 0


__all__ = [
    "record_startup_complete",
    "track_ui_render",
    "track_cache_load",
    "track_auth_latency",
]
