from __future__ import annotations

"""Lightweight performance observability helpers for critical services."""

import csv
import io
import statistics
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Deque, Iterable, Iterator

try:  # pragma: no cover - tracemalloc availability depends on runtime
    import tracemalloc
except ModuleNotFoundError:  # pragma: no cover - fallback for stripped environments
    tracemalloc = None  # type: ignore[assignment]
else:  # pragma: no cover - tracing state is environment specific
    if not tracemalloc.is_tracing():
        tracemalloc.start(25)

from shared.version import __version__
from services.update_checker import record_update_log


_MAX_SAMPLES = 50
_SAMPLES: dict[str, Deque["ExecutionSample"]] = defaultdict(deque)
_LOCK = threading.Lock()


@dataclass(frozen=True)
class ExecutionSample:
    """Represents a single execution measurement."""

    duration_ms: float
    memory_kb: float | None
    timestamp: float
    version: str


@dataclass(frozen=True)
class MetricSummary:
    """Aggregate view of the most recent execution samples."""

    name: str
    samples: int
    average_ms: float
    last_ms: float
    average_memory_kb: float | None
    last_memory_kb: float | None
    last_timestamp: float
    version: str

    @property
    def last_run_iso(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_timestamp))

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "samples": self.samples,
            "average_ms": self.average_ms,
            "last_ms": self.last_ms,
            "average_memory_kb": self.average_memory_kb,
            "last_memory_kb": self.last_memory_kb,
            "last_timestamp": self.last_timestamp,
            "version": self.version,
        }


def _capture_memory() -> int | None:
    if tracemalloc is None:
        return None
    try:
        current, _peak = tracemalloc.get_traced_memory()
    except RuntimeError:  # pragma: no cover - tracing disabled at runtime
        return None
    return int(current)


def record_execution(name: str, duration_ms: float, memory_kb: float | None = None) -> None:
    """Store a single execution sample and mirror it into the update log."""

    sample = ExecutionSample(
        duration_ms=float(duration_ms),
        memory_kb=float(memory_kb) if memory_kb is not None else None,
        timestamp=time.time(),
        version=__version__,
    )
    with _LOCK:
        buffer = _SAMPLES[name]
        buffer.append(sample)
        if len(buffer) > _MAX_SAMPLES:
            buffer.popleft()

    status_parts = [f"duration={sample.duration_ms:.2f}ms"]
    if sample.memory_kb is not None:
        status_parts.append(f"memory={sample.memory_kb:.2f}KB")
    status = " | ".join(status_parts)
    try:
        record_update_log(f"perf:{name}", status, version=sample.version)
    except Exception:  # pragma: no cover - logging failures should not block execution
        pass


def _summarise_samples(name: str, samples: Iterable[ExecutionSample]) -> MetricSummary:
    buffer = list(samples)
    durations = [sample.duration_ms for sample in buffer]
    memories = [sample.memory_kb for sample in buffer if sample.memory_kb is not None]
    last_sample = buffer[-1]
    average_ms = statistics.fmean(durations) if durations else 0.0
    average_memory = statistics.fmean(memories) if memories else None
    return MetricSummary(
        name=name,
        samples=len(buffer),
        average_ms=average_ms,
        last_ms=last_sample.duration_ms,
        average_memory_kb=average_memory,
        last_memory_kb=last_sample.memory_kb,
        last_timestamp=last_sample.timestamp,
        version=last_sample.version,
    )


def get_recent_metrics() -> list[MetricSummary]:
    """Return aggregated performance metrics for monitored functions."""

    with _LOCK:
        items = [
            _summarise_samples(name, tuple(buffer))
            for name, buffer in _SAMPLES.items()
            if buffer
        ]
    return sorted(items, key=lambda item: item.name)


def export_metrics_csv() -> str:
    """Serialise recent metrics to CSV for QA analysis."""

    summaries = get_recent_metrics()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "name",
            "samples",
            "average_ms",
            "last_ms",
            "average_memory_kb",
            "last_memory_kb",
            "last_run",
            "version",
        ]
    )
    for summary in summaries:
        writer.writerow(
            [
                summary.name,
                summary.samples,
                f"{summary.average_ms:.4f}",
                f"{summary.last_ms:.4f}",
                "" if summary.average_memory_kb is None else f"{summary.average_memory_kb:.4f}",
                "" if summary.last_memory_kb is None else f"{summary.last_memory_kb:.4f}",
                summary.last_run_iso,
                summary.version,
            ]
        )
    return output.getvalue()


@contextmanager
def measure_execution(name: str) -> Iterator[None]:
    """Context manager that records execution time and optional memory usage."""

    start = time.perf_counter()
    memory_before = _capture_memory()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        memory_after = _capture_memory()
        memory_delta = None
        if memory_before is not None and memory_after is not None:
            memory_delta = max(0.0, float(memory_after - memory_before) / 1024.0)
        record_execution(name, elapsed_ms, memory_delta)


def track_function(name: str):
    """Decorator that instruments the wrapped callable for performance metrics."""

    def _decorator(func):
        @wraps(func)
        def _wrapped(*args, **kwargs):
            with measure_execution(name):
                return func(*args, **kwargs)

        return _wrapped

    return _decorator


def reset_metrics() -> None:
    """Clear collected metrics (mainly intended for testing)."""

    with _LOCK:
        _SAMPLES.clear()


__all__ = [
    "ExecutionSample",
    "MetricSummary",
    "measure_execution",
    "record_execution",
    "track_function",
    "get_recent_metrics",
    "export_metrics_csv",
    "reset_metrics",
]

