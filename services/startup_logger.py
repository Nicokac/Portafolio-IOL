"""Startup logging utilities for capturing initialization failures early."""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from queue import Full, Queue
from typing import Final, NamedTuple

from shared.version import __version__

_LOGGER_NAME: Final[str] = "app.startup"
_LOG_PATH: Final[Path] = Path("logs") / "app_startup.log"
_QUEUE_MAXSIZE: Final[int] = 256


class _LogTask(NamedTuple):
    level: str
    msg: str
    args: tuple[object, ...]
    kwargs: dict[str, object]


_STOP_SENTINEL: Final[object] = object()
_LOG_QUEUE: Queue[_LogTask | object] = Queue(maxsize=_QUEUE_MAXSIZE)
_WORKER_LOCK = threading.Lock()
_WORKER_STARTED = False


def _ensure_logger_configured() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(_LOG_PATH, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _log_worker() -> None:
    logger = _ensure_logger_configured()
    while True:
        task = _LOG_QUEUE.get()
        try:
            if task is _STOP_SENTINEL:
                for handler in logger.handlers:
                    try:
                        handler.flush()
                    except Exception:
                        logging.getLogger(__name__).debug("Startup logger handler flush failed", exc_info=True)
                return
            level = getattr(logger, task.level, None)
            if level is None:
                continue
            level(task.msg, *task.args, **task.kwargs)
        except Exception:
            logging.getLogger(__name__).debug("Startup logger worker could not persist a record", exc_info=True)
        finally:
            _LOG_QUEUE.task_done()
    # pragma: no cover - graceful shutdown path


def _start_worker_if_needed() -> None:
    global _WORKER_STARTED
    if _WORKER_STARTED:
        return
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        worker = threading.Thread(
            target=_log_worker,
            name="startup-logger-worker",
            daemon=True,
        )
        worker.start()
        _WORKER_STARTED = True


def _submit_log(level: str, msg: str, *args: object, **kwargs: object) -> None:
    _start_worker_if_needed()
    task = _LogTask(level=level, msg=msg, args=args, kwargs=kwargs)
    try:
        _LOG_QUEUE.put_nowait(task)
    except Full:
        logger = _ensure_logger_configured()
        getattr(logger, level, logger.info)(msg, *args, **kwargs)


def _shutdown_worker() -> None:
    global _WORKER_STARTED
    if not _WORKER_STARTED:
        return
    try:
        _LOG_QUEUE.put_nowait(_STOP_SENTINEL)
    except Full:
        _LOG_QUEUE.put(_STOP_SENTINEL)
    _LOG_QUEUE.join()
    logger = logging.getLogger(_LOGGER_NAME)
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            logging.getLogger(__name__).debug("Startup logger handler flush failed", exc_info=True)
    _WORKER_STARTED = False


atexit.register(_shutdown_worker)


def flush_startup_logger() -> None:
    """Block until all pending startup log records are persisted."""

    if not _WORKER_STARTED:
        return
    _LOG_QUEUE.join()
    logger = logging.getLogger(_LOGGER_NAME)
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:
            logging.getLogger(__name__).debug("Startup logger handler flush failed", exc_info=True)


def log_startup_event(message: str) -> None:
    """Record a structured startup event with process and version context."""

    _submit_log("info", "%s | pid=%s | version=%s", message, os.getpid(), __version__)


def log_startup_exception(exc: Exception) -> None:
    """Persist an unexpected startup exception with full traceback."""

    _submit_log(
        "error",
        "Startup exception captured | pid=%s | version=%s | error=%s",
        os.getpid(),
        __version__,
        exc,
    )
    _submit_log("error", "Traceback:\n%s", traceback.format_exc())


def log_ui_total_load_metric(total_ms: float | int | None, *, timestamp: datetime | None = None) -> None:
    """Record the total UI load metric with contextual metadata."""

    ts = timestamp or datetime.now(timezone.utc)
    ts_utc = ts.astimezone(timezone.utc)
    payload = {
        "metric": "ui_total_load",
        "value_ms": None if total_ms is None else float(total_ms),
        "version": __version__,
        "timestamp": ts_utc.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _submit_log("info", json.dumps(payload, ensure_ascii=False))


__all__ = [
    "log_startup_event",
    "log_startup_exception",
    "log_ui_total_load_metric",
    "flush_startup_logger",
]
