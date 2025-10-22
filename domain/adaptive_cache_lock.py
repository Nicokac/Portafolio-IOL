"""Global adaptive cache lock with contention diagnostics."""

from __future__ import annotations

import inspect
import logging
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from types import TracebackType

LOGGER = logging.getLogger(__name__)

_PROLONGED_HOLD_THRESHOLD = 120.0


def _resolve_caller_module() -> str:
    """Best-effort resolution of the caller module."""

    frame = inspect.currentframe()
    if frame is None:  # pragma: no cover - extremely rare on PyPy/Stackless
        return __name__
    frame = frame.f_back
    while frame is not None:
        module = frame.f_globals.get("__name__")
        if module and module != __name__ and not module.startswith("contextlib"):
            return str(module)
        frame = frame.f_back
    return __name__


def _format_owner_label(thread_id: int | None, thread_name: str | None) -> str:
    name = str(thread_name or "thread")
    ident = f"#{thread_id}" if thread_id is not None else ""
    return f"{name}{ident}"


class AdaptiveCacheLock(AbstractContextManager["AdaptiveCacheLock"]):
    """Context manager that wraps a global ``threading.Lock`` with telemetry."""

    def __init__(self, *, warn_after: float = 45.0) -> None:
        self._lock = threading.Lock()
        self._warn_after = float(max(warn_after, 0.0))
        self._owner: int | None = None
        self._owner_name: str | None = None
        self._owner_module: str | None = None
        self._depth = 0
        self._wait_started: float | None = None
        self._acquired_at: float | None = None
        self._last_wait_time: float = 0.0
        self._last_hold_time: float = 0.0
        self._released_early = False

    # ------------------------------------------------------------------
    # Internal acquisition helpers
    # ------------------------------------------------------------------
    def _acquire(self, *, timeout: float | None) -> bool:
        thread_id = threading.get_ident()
        if self._owner == thread_id:
            self._depth += 1
            return True

        caller_module = _resolve_caller_module()
        current_thread = threading.current_thread()
        thread_name = getattr(current_thread, "name", None)
        self._wait_started = time.monotonic()

        if timeout is None:
            acquired = self._lock.acquire()
        else:
            acquired = self._lock.acquire(timeout=max(timeout, 0.0))
        if not acquired:
            self._wait_started = None
            return False

        acquired_at = time.monotonic()
        self._owner = thread_id
        self._owner_name = thread_name
        self._owner_module = caller_module
        self._depth = 1
        self._acquired_at = acquired_at
        self._last_hold_time = 0.0
        self._last_wait_time = 0.0
        self._released_early = False

        if self._warn_after > 0 and self._wait_started is not None:
            waited = acquired_at - self._wait_started
            self._last_wait_time = max(waited, 0.0)
            if waited > self._warn_after:
                owner_label = _format_owner_label(thread_id, thread_name)
                LOGGER.warning(
                    (
                        "El lock adaptativo demoró %.2fs en adquirirse "
                        "(umbral %.2fs, módulo=%s, owner=%s)"
                    ),
                    waited,
                    self._warn_after,
                    caller_module,
                    owner_label,
                )
                self._record_lock_wait_event(
                    wait_time=max(waited, 0.0),
                    hold_time=0.0,
                    owner_label=owner_label,
                    thread_name=thread_name,
                    module_name=caller_module,
                    reason="wait",
                )

        return True

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> "AdaptiveCacheLock":
        self._acquire(timeout=None)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        thread_id = threading.get_ident()
        if self._owner != thread_id:
            if self._released_early:
                # Lock was released by ``run_in_background``.
                self._released_early = False
                return None
            raise RuntimeError(
                "El lock adaptativo fue liberado por un hilo no propietario"
            )

        if self._depth > 1:
            self._depth -= 1
            return None

        owner_label = _format_owner_label(self._owner, self._owner_name)
        module_name = self._owner_module or _resolve_caller_module()
        thread_name = self._owner_name
        wait_time = self._last_wait_time
        held_for = 0.0

        try:
            if self._acquired_at is not None:
                held_for = max(time.monotonic() - self._acquired_at, 0.0)
                self._last_hold_time = held_for
                if self._warn_after > 0 and held_for > self._warn_after:
                    LOGGER.warning(
                        (
                            "El lock adaptativo permaneció retenido %.2fs "
                            "(umbral %.2fs, módulo=%s, owner=%s)"
                        ),
                        held_for,
                        self._warn_after,
                        module_name,
                        owner_label,
                    )
                if held_for > _PROLONGED_HOLD_THRESHOLD:
                    LOGGER.warning(
                        (
                            "Retención prolongada del lock adaptativo: %.2fs "
                            "(módulo=%s, owner=%s)"
                        ),
                        held_for,
                        module_name,
                        owner_label,
                    )
        finally:
            self._owner = None
            self._owner_name = None
            self._owner_module = None
            self._depth = 0
            self._wait_started = None
            self._acquired_at = None
            self._lock.release()
            self._released_early = False

        self._record_lock_wait_event(
            wait_time=wait_time,
            hold_time=held_for,
            owner_label=owner_label,
            thread_name=thread_name,
            module_name=module_name,
            reason="hold",
        )
        return None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def acquire_with_timeout(self, timeout: float) -> bool:
        """Attempt to acquire the lock within ``timeout`` seconds."""

        timeout = max(float(timeout), 0.0)
        return self._acquire(timeout=timeout)

    def release(self) -> None:
        """Release the lock acquired with :meth:`acquire_with_timeout`."""

        self.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def locked(self) -> bool:
        """Expose whether the underlying lock is currently held."""

        return self._owner is not None

    def run_in_background(
        self,
        target: Callable[..., object],
        /,
        *args,
        name: str | None = None,
        daemon: bool = True,
        **kwargs,
    ) -> threading.Thread:
        """Release the lock early and schedule ``target`` to run asynchronously.

        This helper is intended for expensive predictive operations that should
        not keep the adaptive cache locked.  It can only be invoked by the
        thread that currently owns the lock at the top-most acquisition level.
        The lock is released before the worker thread starts so that other
        operations can proceed immediately.
        """

        thread_id = threading.get_ident()
        if self._owner != thread_id:
            raise RuntimeError(
                "Solo el propietario del lock puede delegar trabajo en segundo plano"
            )
        if self._depth != 1:
            raise RuntimeError(
                "No se puede delegar trabajo en segundo plano con "
                "adquisiciones reentrantes"
            )

        owner_label = _format_owner_label(self._owner, self._owner_name)
        module_name = self._owner_module or _resolve_caller_module()
        thread_name = self._owner_name
        wait_time = self._last_wait_time
        held_for = 0.0
        if self._acquired_at is not None:
            held_for = max(time.monotonic() - self._acquired_at, 0.0)
            if self._warn_after > 0 and held_for > self._warn_after:
                LOGGER.warning(
                    (
                        "El lock adaptativo permaneció retenido %.2fs antes de "
                        "delegar (umbral %.2fs, módulo=%s, owner=%s)"
                    ),
                    held_for,
                    self._warn_after,
                    module_name,
                    owner_label,
                )
        self._owner = None
        self._owner_name = None
        self._owner_module = None
        self._depth = 0
        self._wait_started = None
        self._acquired_at = None
        self._released_early = True
        self._lock.release()

        self._record_lock_wait_event(
            wait_time=wait_time,
            hold_time=held_for,
            owner_label=owner_label,
            thread_name=thread_name,
            module_name=module_name,
            reason="delegate",
        )

        return run_in_background(
            target,
            *args,
            name=name,
            daemon=daemon,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_lock_wait_event(
        self,
        *,
        wait_time: float,
        hold_time: float,
        owner_label: str,
        thread_name: str | None,
        module_name: str,
        reason: str,
    ) -> None:
        exceeds_wait = self._warn_after > 0 and wait_time > self._warn_after
        exceeds_hold = hold_time > _PROLONGED_HOLD_THRESHOLD
        if not exceeds_wait and not exceeds_hold:
            return
        try:
            from services.performance_timer import record_stage
        except Exception:  # pragma: no cover - optional dependency failures
            LOGGER.debug(
                "No se pudo registrar lock_wait en performance_timer",
                exc_info=True,
            )
            return

        payload = {
            "module": module_name,
            "lock_owner": owner_label,
            "wait_time_s": f"{max(wait_time, 0.0):.3f}",
            "held_time_s": f"{max(hold_time, 0.0):.3f}",
            "thread_name": thread_name or owner_label,
            "reason": reason,
        }
        try:
            record_stage(
                "lock_wait",
                total_ms=max(wait_time, hold_time, 0.0) * 1000.0,
                extra=payload,
            )
        except Exception:  # pragma: no cover - logging infrastructure failures
            LOGGER.debug("Fallo al registrar lock_wait", exc_info=True)


def run_in_background(
    target: Callable[..., object],
    /,
    *args,
    name: str | None = None,
    daemon: bool = True,
    logger: logging.Logger | None = None,
    **kwargs,
) -> threading.Thread:
    """Execute ``target`` in a background thread with instrumentation logs."""

    thread_logger = logger or LOGGER
    target_name = getattr(target, "__name__", repr(target))
    thread_name = name or f"bg-{target_name}-{int(time.time() * 1000) % 10000}"

    def _runner() -> None:
        worker = threading.current_thread()
        thread_logger.info(
            "Iniciando tarea en segundo plano '%s' (hilo=%s)",
            target_name,
            worker.name,
        )
        started = time.perf_counter()
        try:
            target(*args, **kwargs)
        except Exception:  # pragma: no cover - propagated to logs for observability
            thread_logger.exception(
                "Error en tarea en segundo plano '%s' (hilo=%s)",
                target_name,
                worker.name,
            )
        finally:
            elapsed = time.perf_counter() - started
            thread_logger.info(
                "Tarea en segundo plano '%s' finalizada en %.2fs (hilo=%s)",
                target_name,
                elapsed,
                worker.name,
            )

    worker = threading.Thread(target=_runner, name=thread_name, daemon=daemon)
    worker.start()
    return worker


adaptive_cache_lock = AdaptiveCacheLock()


__all__ = ["adaptive_cache_lock", "AdaptiveCacheLock", "run_in_background"]
