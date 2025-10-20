"""Token bucket rate limiter used to throttle expensive operations."""

from __future__ import annotations

import time
from threading import Lock
from typing import Callable

__all__ = ["RateLimiter"]


class RateLimiter:
    """Thread-safe token bucket implementation."""

    def __init__(
        self,
        *,
        capacity: int,
        refill_rate: float,
        monotonic: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be greater than zero")
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._refill_rate = float(refill_rate)
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleeper or time.sleep
        self._lock = Lock()
        self._last_refill = self._monotonic()

    def acquire(self, tokens: float = 1.0) -> None:
        """Block until ``tokens`` are available in the bucket."""

        if tokens <= 0:
            return

        request = float(tokens)
        while True:
            with self._lock:
                now = self._monotonic()
                elapsed = now - self._last_refill
                if elapsed > 0:
                    self._tokens = min(
                        self._capacity,
                        self._tokens + elapsed * self._refill_rate,
                    )
                    self._last_refill = now
                if self._tokens >= request:
                    self._tokens -= request
                    return
                needed = request - self._tokens
                wait_time = needed / self._refill_rate

            self._sleep(wait_time)
