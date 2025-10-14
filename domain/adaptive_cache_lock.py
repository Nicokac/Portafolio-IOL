"""Global adaptive cache lock with contention diagnostics."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import AbstractContextManager
from typing import Callable, Optional

LOGGER = logging.getLogger(__name__)


class AdaptiveCacheLock(AbstractContextManager["AdaptiveCacheLock"]):
    """Context manager that wraps a global ``threading.Lock`` with telemetry."""

    def __init__(self, *, warn_after: float = 45.0) -> None:
        self._lock = threading.Lock()
        self._warn_after = float(max(warn_after, 0.0))
        self._owner: Optional[int] = None
        self._depth = 0
        self._wait_started: float | None = None
        self._acquired_at: float | None = None
        self._released_early = False

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> "AdaptiveCacheLock":
        thread_id = threading.get_ident()
        if self._owner == thread_id:
            # Re-entrant acquisition from the same thread; increase depth only.
            self._depth += 1
            return self

        self._wait_started = time.monotonic()
        self._lock.acquire()
        acquired_at = time.monotonic()
        self._owner = thread_id
        self._depth = 1
        self._acquired_at = acquired_at
        self._released_early = False

        if self._warn_after > 0 and self._wait_started is not None:
            waited = acquired_at - self._wait_started
            if waited > self._warn_after:
                LOGGER.warning(
                    "El lock adaptativo demoró %.2fs en adquirirse (umbral %.2fs)",
                    waited,
                    self._warn_after,
                )

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        thread_id = threading.get_ident()
        if self._owner != thread_id:
            if self._released_early:
                # Lock was released by ``run_in_background``.
                self._released_early = False
                return None
            raise RuntimeError("El lock adaptativo fue liberado por un hilo no propietario")

        if self._depth > 1:
            self._depth -= 1
            return None

        held_for = 0.0
        if self._acquired_at is not None:
            held_for = time.monotonic() - self._acquired_at
            if self._warn_after > 0 and held_for > self._warn_after:
                LOGGER.warning(
                    "El lock adaptativo permaneció retenido %.2fs (umbral %.2fs)",
                    held_for,
                    self._warn_after,
                )

        self._owner = None
        self._depth = 0
        self._wait_started = None
        self._acquired_at = None
        self._lock.release()
        self._released_early = False
        return None

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
                "No se puede delegar trabajo en segundo plano con adquisiciones reentrantes"
            )

        if self._acquired_at is not None and self._warn_after > 0:
            held_for = time.monotonic() - self._acquired_at
            if held_for > self._warn_after:
                LOGGER.warning(
                    "El lock adaptativo permaneció retenido %.2fs antes de delegar (umbral %.2fs)",
                    held_for,
                    self._warn_after,
                )

        self._owner = None
        self._depth = 0
        self._wait_started = None
        self._acquired_at = None
        self._released_early = True
        self._lock.release()

        worker = threading.Thread(
            target=target,
            name=name,
            args=args,
            kwargs=kwargs,
            daemon=daemon,
        )
        worker.start()
        return worker


adaptive_cache_lock = AdaptiveCacheLock()


__all__ = ["adaptive_cache_lock", "AdaptiveCacheLock"]
