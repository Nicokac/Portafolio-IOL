"""Lightweight timing decorator to build CSV timelines for UI profiling."""

from __future__ import annotations

import csv
import os
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

from .rerun_trace import mark_event
from .ui_flow import current_flow_id, ensure_flow_id

P = ParamSpec("P")
T = TypeVar("T")

DEBUG_TIMELINE = os.getenv("DEBUG_TIMELINE", "0") == "1"
_PERF_DIR = Path("perf")
_FIELDNAMES = (
    "timestamp",
    "flow_id",
    "function",
    "duration_ms",
    "status",
    "thread",
)


def _resolve_flow_id() -> str:
    flow_id = current_flow_id()
    if flow_id:
        return flow_id
    return ensure_flow_id()


def _write_row(row: dict[str, Any]) -> None:
    _PERF_DIR.mkdir(parents=True, exist_ok=True)
    flow_id = row.get("flow_id", "no-session")
    path = _PERF_DIR / f"timings_{flow_id}.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def timeit(label: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorate ``func`` to record execution timings when enabled."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if not DEBUG_TIMELINE:
            return func

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            flow_id = _resolve_flow_id()
            start = time.perf_counter()
            mark_event("timeit_start", label, {"flow_id": flow_id})
            status = "ok"
            try:
                return func(*args, **kwargs)
            except Exception:
                status = "error"
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000.0
                row = {
                    "timestamp": time.time(),
                    "flow_id": flow_id,
                    "function": label,
                    "duration_ms": f"{duration_ms:.3f}",
                    "status": status,
                    "thread": threading.current_thread().name,
                }
                mark_event(
                    "timeit_end",
                    label,
                    {"flow_id": flow_id, "duration_ms": duration_ms, "status": status},
                )
                _write_row(row)

        return wrapper

    return decorator


__all__ = ["DEBUG_TIMELINE", "timeit"]
