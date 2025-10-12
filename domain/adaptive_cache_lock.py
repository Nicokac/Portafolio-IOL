"""Global adaptive cache lock with contention diagnostics."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import AbstractContextManager
from typing import Optional

LOGGER = logging.getLogger(__name__)


class AdaptiveCacheLock(AbstractContextManager["AdaptiveCacheLock"]):
    """Context manager that wraps a global ``threading.Lock`` with telemetry."""

    def __init__(self, *, warn_after: float = 5.0) -> None:
        self._lock = threading.Lock()
        self._warn_after = float(max(warn_after, 0.0))
        self._owner: Optional[int] = None
        self._depth = 0
        self._wait_started: float | None = None
        self._acquired_at: float | None = None

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

        if self._warn_after > 0 and self._wait_started is not None:
            waited = acquired_at - self._wait_started
            if waited > self._warn_after:
                LOGGER.warning(
                    "El lock adaptativo demoró %.2fs en adquirirse", waited
                )

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        thread_id = threading.get_ident()
        if self._owner != thread_id:
            raise RuntimeError("El lock adaptativo fue liberado por un hilo no propietario")

        if self._depth > 1:
            self._depth -= 1
            return None

        held_for = 0.0
        if self._acquired_at is not None:
            held_for = time.monotonic() - self._acquired_at
            if self._warn_after > 0 and held_for > self._warn_after:
                LOGGER.warning(
                    "El lock adaptativo permaneció retenido %.2fs", held_for
                )

        self._owner = None
        self._depth = 0
        self._wait_started = None
        self._acquired_at = None
        self._lock.release()
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def locked(self) -> bool:
        """Expose whether the underlying lock is currently held."""

        return self._owner is not None


adaptive_cache_lock = AdaptiveCacheLock()


__all__ = ["adaptive_cache_lock", "AdaptiveCacheLock"]
