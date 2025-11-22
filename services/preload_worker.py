"""Lazy scientific preload worker orchestrated around the login flow."""

from __future__ import annotations

import importlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Iterable

from datetime import datetime, timezone

from services.startup_logger import log_startup_event


class PreloadPhase(str, Enum):
    """Execution state for the preload worker."""

    IDLE = "idle"
    PAUSED = "paused"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class PreloadTelemetry:
    """Runtime measurements for the preload execution."""

    durations_ms: dict[str, float | None] = field(default_factory=dict)
    total_ms: float | None = None
    status: PreloadPhase = PreloadPhase.IDLE
    error: str | None = None
    timestamp: str | None = None


_DEFAULT_LIBRARIES: tuple[str, ...] = ("pandas", "plotly", "statsmodels")
_THREAD_NAME = "preload-libraries-worker"

_logger = logging.getLogger(__name__)

_WORKER_LOCK = threading.Lock()
_RESUME_EVENT = threading.Event()
_FINISHED_EVENT = threading.Event()
_WORKER_THREAD: threading.Thread | None = None
_LIBRARY_OVERRIDE: tuple[str, ...] | None = None
_TELEMETRY = PreloadTelemetry()
_PHASE: PreloadPhase = PreloadPhase.IDLE


def _format_duration_ms(duration: float) -> float:
    return round(duration * 1000.0, 2)


@lru_cache(maxsize=1)
def _get_metric_updaters():
    from services.performance_timer import (
        update_preload_library_metric,
        update_preload_total_metric,
    )

    return update_preload_total_metric, update_preload_library_metric


def _iter_libraries(custom_libraries: Iterable[str] | None = None) -> tuple[str, ...]:
    if custom_libraries is not None:
        return tuple(custom_libraries)
    if _LIBRARY_OVERRIDE is not None:
        return _LIBRARY_OVERRIDE
    env_override = os.getenv("APP_PRELOAD_LIBS")
    if env_override:
        return tuple(lib.strip() for lib in env_override.split(",") if lib.strip())
    return _DEFAULT_LIBRARIES


def _set_phase(phase: PreloadPhase) -> None:
    global _PHASE
    _PHASE = phase
    _TELEMETRY.status = phase


def _reset_events() -> None:
    _RESUME_EVENT.clear()
    _FINISHED_EVENT.clear()


def _log_preload_metric(library: str, duration: float, status: str, error: str | None = None) -> None:
    payload: dict[str, object] = {
        "event": "preload_library",
        "module_name": library,
        "status": status,
        "duration_ms": _format_duration_ms(duration),
        "timestamp": _now_iso(),
    }
    if error:
        payload["error"] = error
    log_startup_event(json.dumps(payload, ensure_ascii=False))


def _log_preload_total(
    *, status: str, total_ms: float, libraries: tuple[str, ...], resume_delay_ms: float, error: str | None
) -> None:
    payload: dict[str, object] = {
        "event": "preload_total",
        "module_name": "all",
        "status": status,
        "duration_ms": round(total_ms, 2),
        "resume_delay_ms": round(resume_delay_ms, 2),
        "libraries": list(libraries),
        "timestamp": _now_iso(),
    }
    if error:
        payload["error"] = error
    log_startup_event(json.dumps(payload, ensure_ascii=False))


def _update_library_metric(library: str, duration_ms: float | None) -> None:
    try:
        _, update_library = _get_metric_updaters()
    except Exception:  # pragma: no cover - optional dependency wiring
        return
    try:
        update_library(library, duration_ms)
    except Exception:  # pragma: no cover - metrics backend failure
        _logger.debug("No se pudo actualizar la métrica de precarga %s", library, exc_info=True)


def _update_total_metric(total_ms: float | None) -> None:
    try:
        update_total, _ = _get_metric_updaters()
    except Exception:  # pragma: no cover - optional dependency wiring
        return
    try:
        update_total(total_ms)
    except Exception:  # pragma: no cover - metrics backend failure
        _logger.debug("No se pudo actualizar la métrica preload_total_ms", exc_info=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _preload_target(libraries: Iterable[str] | None = None) -> None:
    global _WORKER_THREAD
    libraries_tuple = _iter_libraries(tuple(libraries) if libraries is not None else None)
    _TELEMETRY.timestamp = _now_iso()
    log_startup_event(
        json.dumps(
            {
                "event": "preload_waiting_resume",
                "libraries": list(libraries_tuple),
                "timestamp": _TELEMETRY.timestamp,
            }
        )
    )
    _set_phase(PreloadPhase.PAUSED)
    resume_started = time.perf_counter()
    _RESUME_EVENT.wait()
    resume_delay_ms = (time.perf_counter() - resume_started) * 1000.0
    _set_phase(PreloadPhase.RUNNING)
    log_startup_event(
        json.dumps(
            {
                "event": "preload_running",
                "libraries": list(libraries_tuple),
                "resume_delay_ms": round(resume_delay_ms, 2),
                "timestamp": _now_iso(),
            },
            ensure_ascii=False,
        )
    )
    start_total = time.perf_counter()
    durations: dict[str, float | None] = {}
    first_error: str | None = None
    try:
        for library in libraries_tuple:
            start = time.perf_counter()
            status = "success"
            error_message: str | None = None
            try:
                importlib.import_module(library)
            except Exception as exc:  # pragma: no cover - depends on optional libs
                status = "error"
                error_message = repr(exc)
                if first_error is None:
                    first_error = error_message
                _logger.debug("Preload de %s falló", library, exc_info=True)
            elapsed = time.perf_counter() - start
            duration_ms = elapsed * 1000.0
            durations[library] = duration_ms if status == "success" else None
            _log_preload_metric(library, elapsed, status, error_message)
            _update_library_metric(library, duration_ms if status == "success" else float("nan"))
        total_ms = (time.perf_counter() - start_total) * 1000.0
    except Exception as exc:  # pragma: no cover - catastrophic failure
        total_ms = (time.perf_counter() - start_total) * 1000.0
        first_error = first_error or repr(exc)
        _logger.debug("El worker de precarga finalizó con error inesperado", exc_info=True)
        _set_phase(PreloadPhase.FAILED)
    else:
        status_value = "completed" if first_error is None else "completed_with_errors"
        _set_phase(PreloadPhase.COMPLETED if first_error is None else PreloadPhase.FAILED)
    finally:
        _TELEMETRY.durations_ms = durations
        _TELEMETRY.total_ms = total_ms
        _TELEMETRY.status = _PHASE
        _TELEMETRY.error = first_error
        _log_preload_total(
            status=_PHASE.value,
            total_ms=total_ms,
            libraries=libraries_tuple,
            resume_delay_ms=resume_delay_ms,
            error=first_error,
        )
        _update_total_metric(total_ms)
        _FINISHED_EVENT.set()
        _WORKER_THREAD = None


def start_preload_worker(
    libraries: Iterable[str] | None = None,
    *,
    paused: bool = True,
) -> bool:
    """Start the preload worker thread once."""

    global _LIBRARY_OVERRIDE, _WORKER_THREAD

    with _WORKER_LOCK:
        if _WORKER_THREAD and _WORKER_THREAD.is_alive():
            if libraries is not None:
                _LIBRARY_OVERRIDE = tuple(libraries)
            if not paused and not _RESUME_EVENT.is_set():
                _RESUME_EVENT.set()
            return False

        _LIBRARY_OVERRIDE = tuple(libraries) if libraries is not None else None
        _reset_events()
        _TELEMETRY.durations_ms = {}
        _TELEMETRY.total_ms = None
        _TELEMETRY.status = PreloadPhase.PAUSED
        _TELEMETRY.error = None
        _TELEMETRY.timestamp = None
        _set_phase(PreloadPhase.PAUSED if paused else PreloadPhase.RUNNING)

        thread = threading.Thread(
            target=_preload_target,
            args=(libraries,),
            name=_THREAD_NAME,
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            _logger.debug("No se pudo iniciar el preload worker", exc_info=True)
            return False
        _WORKER_THREAD = thread
        if not paused:
            _RESUME_EVENT.set()
        return True


def resume_preload_worker(
    *,
    delay_seconds: float = 0.0,
    libs_override: Iterable[str] | None = None,
    libraries: Iterable[str] | None = None,
) -> bool:
    """Signal the worker to resume the import sequence.

    ``libs_override`` allows callers to customize the modules to import at resume time
    without blocking the main thread. The optional ``libraries`` parameter is kept for
    backwards compatibility with older call sites.
    """

    global _LIBRARY_OVERRIDE

    override = libs_override if libs_override is not None else libraries

    with _WORKER_LOCK:
        if _WORKER_THREAD is None or not _WORKER_THREAD.is_alive():
            started = start_preload_worker(libraries=override, paused=True)
            if not started:
                return False
        elif override is not None:
            _LIBRARY_OVERRIDE = tuple(override)

        if _RESUME_EVENT.is_set():
            return False

        def _resume() -> None:
            _RESUME_EVENT.set()

        if delay_seconds > 0:
            timer = threading.Timer(delay_seconds, _resume)
            timer.daemon = True
            timer.start()
        else:
            _resume()
        return True


def is_preload_complete() -> bool:
    return _FINISHED_EVENT.is_set()


def wait_for_preload_completion(timeout: float | None = None) -> bool:
    return _FINISHED_EVENT.wait(timeout)


def get_preload_status() -> PreloadPhase:
    return _PHASE


def get_preload_metrics() -> PreloadTelemetry:
    return PreloadTelemetry(
        durations_ms=dict(_TELEMETRY.durations_ms),
        total_ms=_TELEMETRY.total_ms,
        status=_TELEMETRY.status,
        error=_TELEMETRY.error,
        timestamp=_TELEMETRY.timestamp,
    )


def reset_worker_for_tests() -> None:
    """Reset global state for deterministic unit tests."""

    global _LIBRARY_OVERRIDE, _WORKER_THREAD

    with _WORKER_LOCK:
        if _WORKER_THREAD and _WORKER_THREAD.is_alive():
            raise RuntimeError("Cannot reset preload worker while it is running")
        _LIBRARY_OVERRIDE = None
        _reset_events()
        _TELEMETRY.durations_ms = {}
        _TELEMETRY.total_ms = None
        _TELEMETRY.status = PreloadPhase.IDLE
        _TELEMETRY.error = None
        _set_phase(PreloadPhase.IDLE)


__all__ = [
    "PreloadPhase",
    "PreloadTelemetry",
    "get_preload_metrics",
    "get_preload_status",
    "is_preload_complete",
    "reset_worker_for_tests",
    "resume_preload_worker",
    "start_preload_worker",
    "wait_for_preload_completion",
]
