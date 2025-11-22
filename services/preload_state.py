"""Shared preload completion flag and future subscription helpers."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, TimeoutError
from typing import Callable

_logger = logging.getLogger(__name__)

_preload_done: bool = False
_preload_future: Future[bool] = Future()
_lock = threading.Lock()


def _new_future() -> Future[bool]:
    future: Future[bool] = Future()
    return future


def mark_preload_pending() -> None:
    """Reset the preload completion flag and replace the shared future."""

    global _preload_done, _preload_future
    with _lock:
        _preload_done = False
        _preload_future = _new_future()


def mark_preload_done(success: bool = True) -> None:
    """Mark the preload sequence as finished and resolve subscribers."""

    global _preload_done
    with _lock:
        _preload_done = success
        future = _preload_future
    if not future.done():
        try:
            future.set_result(success)
        except Exception:  # pragma: no cover - defensive fallback
            _logger.debug("No se pudo resolver el futuro de precarga", exc_info=True)


def is_preload_done() -> bool:
    with _lock:
        return _preload_done


def wait_for_preload_ready(timeout: float | None = None) -> bool:
    """Wait for the preload future to complete without blocking indefinitely."""

    future = _preload_future
    try:
        return bool(future.result(timeout=timeout))
    except TimeoutError:
        return False
    except Exception:  # pragma: no cover - defensive fallback
        _logger.debug("No se pudo esperar el futuro de precarga", exc_info=True)
        return False


def subscribe_preload_completion(callback: Callable[[bool], None]) -> None:
    """Register a callback to execute once the preload future resolves."""

    future = _preload_future

    def _run_callback(done_future: Future[bool]) -> None:
        try:
            result = bool(done_future.result())
        except Exception:
            result = False
        try:
            callback(result)
        except Exception:  # pragma: no cover - defensive fallback
            _logger.debug("Suscriptor de precarga falló", exc_info=True)

    if future.done():
        _run_callback(future)
        return

    try:
        future.add_done_callback(_run_callback)
    except Exception:  # pragma: no cover - defensive fallback
        _logger.debug("No se pudo agregar callback al futuro de precarga", exc_info=True)


def reset_preload_state_for_tests() -> None:
    mark_preload_pending()


__all__ = [
    "is_preload_done",
    "mark_preload_done",
    "mark_preload_pending",
    "reset_preload_state_for_tests",
    "subscribe_preload_completion",
    "wait_for_preload_ready",
]
