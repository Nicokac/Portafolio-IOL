"""Utility helpers shared across macro data providers."""
from __future__ import annotations

import time
from threading import Lock
from typing import Callable


class RateLimiter:
    """Simple rate limiter that spaces out requests at a fixed interval."""

    def __init__(
        self,
        *,
        calls_per_minute: int,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if calls_per_minute < 0:
            raise ValueError("calls_per_minute must be >= 0")
        self._interval = 0.0
        if calls_per_minute:
            self._interval = 60.0 / float(calls_per_minute)
        self._monotonic = monotonic
        self._sleep = sleeper
        self._lock = Lock()
        self._next_time = 0.0

    def acquire(self) -> None:
        """Block until the caller is allowed to make another request."""

        if self._interval <= 0:
            return
        with self._lock:
            now = self._monotonic()
            wait_for = self._next_time - now
            if wait_for > 0:
                self._sleep(wait_for)
                now = self._monotonic()
            self._next_time = now + self._interval


__all__ = ["RateLimiter"]
