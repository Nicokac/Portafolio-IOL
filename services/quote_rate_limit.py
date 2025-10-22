from __future__ import annotations

import time
from threading import Lock
from typing import Dict, Tuple

from services.health import record_quote_rate_limit_wait
from shared.settings import quotes_rps_iol, quotes_rps_legacy


class _TokenBucket:
    """Simple token bucket implementation backed by ``time.monotonic``."""

    def __init__(self, rate: float) -> None:
        self._lock = Lock()
        self._rate = 0.0
        self._capacity = float("inf")
        self._tokens = float("inf")
        self._updated = time.monotonic()
        self.configure(rate)

    def configure(self, rate: float) -> None:
        """Update the bucket rate while preserving accumulated tokens."""

        normalized = max(float(rate), 0.0)
        with self._lock:
            self._rate = normalized
            if normalized <= 0:
                self._capacity = float("inf")
                self._tokens = float("inf")
            else:
                capacity = max(normalized * 2.0, 1.0)
                self._capacity = capacity
                if self._tokens == float("inf"):
                    self._tokens = capacity
                else:
                    self._tokens = min(self._tokens, capacity)
            self._updated = time.monotonic()

    def acquire(self, tokens: float = 1.0) -> float:
        """Return required wait time (in seconds) to consume ``tokens``."""

        with self._lock:
            if self._rate <= 0:
                self._tokens = float("inf")
                self._updated = time.monotonic()
                return 0.0

            now = time.monotonic()
            elapsed = max(0.0, now - self._updated)
            if elapsed > 0:
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._updated = now

            required = max(float(tokens), 0.0)
            if self._tokens >= required:
                self._tokens -= required
                return 0.0

            deficit = required - self._tokens
            wait = deficit / self._rate if self._rate > 0 else 0.0
            self._tokens = 0.0
            return wait

    def drain(self) -> None:
        """Force the bucket to be empty going forward."""

        with self._lock:
            if self._rate <= 0:
                self._tokens = float("inf")
            else:
                self._tokens = 0.0
            self._updated = time.monotonic()


class QuoteRateLimiter:
    """Manage token buckets for quote providers (IOL v2 / Legacy)."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._buckets: Dict[str, _TokenBucket] = {}

    def _resolve_rate(self, provider: str) -> float:
        key = str(provider or "iol").casefold()
        if key == "legacy":
            return max(float(quotes_rps_legacy), 0.0)
        return max(float(quotes_rps_iol), 0.0)

    def _get_bucket(self, provider: str) -> Tuple[_TokenBucket, float]:
        rate = self._resolve_rate(provider)
        key = str(provider or "iol").casefold()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _TokenBucket(rate)
                self._buckets[key] = bucket
            else:
                bucket.configure(rate)
        return bucket, rate

    def wait_for_slot(self, provider: str, *, tokens: float = 1.0) -> float:
        """Block until a slot is available for ``provider`` and return wait time."""

        bucket, _ = self._get_bucket(provider)
        wait_time = bucket.acquire(tokens)
        if wait_time > 0:
            record_quote_rate_limit_wait(provider, wait_time, reason="throttle")
            time.sleep(wait_time)
        return wait_time

    def penalize(self, provider: str, *, minimum_wait: float | None = None) -> float:
        """Drain the bucket and sleep for at least ``minimum_wait`` seconds."""

        bucket, rate = self._get_bucket(provider)
        bucket.drain()
        wait_time = max(float(minimum_wait or 0.0), 0.0)
        if rate > 0:
            wait_time = max(wait_time, 1.0 / rate)
        if wait_time > 0:
            record_quote_rate_limit_wait(provider, wait_time, reason="http_429")
            time.sleep(wait_time)
        return wait_time


quote_rate_limiter = QuoteRateLimiter()


__all__ = ["quote_rate_limiter", "QuoteRateLimiter"]
