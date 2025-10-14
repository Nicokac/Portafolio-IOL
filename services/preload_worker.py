"""Background worker to preload heavyweight libraries during the login screen."""

from __future__ import annotations

import importlib
import logging
import os
import threading
import time
from typing import Iterable

from services.startup_logger import log_startup_event

_LIBRARIES: tuple[str, ...] = ("pandas", "plotly", "statsmodels")
_THREAD_NAME = "preload-libraries-worker"
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()

logger = logging.getLogger(__name__)


def _format_duration_ms(duration: float) -> str:
    return f"{duration * 1000.0:.2f}"


def _log_preload_metric(library: str, duration: float, status: str, error: str | None = None) -> None:
    message = (
        "preload"  # namespace identifier
        f" | library={library}"
        f" | status={status}"
        f" | duration_ms={_format_duration_ms(duration)}"
    )
    if error:
        message += f" | error={error}"
    log_startup_event(message)


def _iter_libraries(custom_libraries: Iterable[str] | None = None) -> Iterable[str]:
    if custom_libraries is not None:
        return tuple(custom_libraries)
    env_override = os.getenv("APP_PRELOAD_LIBS")
    if env_override:
        return tuple(lib.strip() for lib in env_override.split(",") if lib.strip())
    return _LIBRARIES


def _preload_target(libraries: Iterable[str] | None = None) -> None:
    libs = _iter_libraries(libraries)
    for library in libs:
        start = time.perf_counter()
        error_message: str | None = None
        status = "success"
        try:
            importlib.import_module(library)
        except Exception as exc:  # pragma: no cover - depends on optional libs
            status = "error"
            error_message = repr(exc)
            logger.debug("Preload of %s failed", library, exc_info=True)
        finally:
            elapsed = time.perf_counter() - start
            _log_preload_metric(library, elapsed, status, error=error_message)


def start_preload_worker(libraries: Iterable[str] | None = None) -> bool:
    """Start the preload worker once and return ``True`` when spawned."""

    global _WORKER_STARTED
    if _WORKER_STARTED:
        return False
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return False
        thread = threading.Thread(
            target=_preload_target,
            args=(libraries,),
            name=_THREAD_NAME,
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            logger.debug("No se pudo iniciar el preload worker", exc_info=True)
            return False
        _WORKER_STARTED = True
        return True


__all__ = ["start_preload_worker"]
