"""Rate limiting middleware protecting the refresh endpoint."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Callable, Dict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from services.auth import describe_active_token


@dataclass
class BucketState:
    """Simple token bucket state container."""

    tokens: float
    last_refill: float


class TokenBucketRateLimiter:
    """Token bucket rate limiter keeping counters in memory."""

    def __init__(
        self,
        capacity: int,
        refill_rate_per_second: float,
        *,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_rate_per_second <= 0:
            raise ValueError("refill_rate_per_second must be positive")
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate_per_second)
        self.time_source = time_source or time.time
        self._buckets: Dict[str, BucketState] = {}
        self.attempt_counters: Dict[str, int] = {}
        self.total_attempts = 0
        self._lock = asyncio.Lock()

    def set_time_source(self, time_source: Callable[[], float]) -> None:
        """Override the internal time source (useful for tests)."""

        self.time_source = time_source

    def reset(self) -> None:
        """Reset all internal counters and buckets."""

        self._buckets.clear()
        self.attempt_counters.clear()
        self.total_attempts = 0

    def _get_bucket(self, identifier: str, now: float) -> BucketState:
        state = self._buckets.get(identifier)
        if state is None:
            state = BucketState(tokens=self.capacity, last_refill=now)
            self._buckets[identifier] = state
        return state

    def _refill(self, state: BucketState, now: float) -> None:
        elapsed = max(0.0, now - state.last_refill)
        if not elapsed:
            return
        state.tokens = min(self.capacity, state.tokens + elapsed * self.refill_rate)
        state.last_refill = now

    def consume(self, identifier: str) -> bool:
        """Attempt to consume a token for ``identifier``."""

        now = self.time_source()
        state = self._get_bucket(identifier, now)
        self._refill(state, now)
        self.total_attempts += 1
        self.attempt_counters[identifier] = self.attempt_counters.get(identifier, 0) + 1
        if state.tokens < 1:
            return False
        state.tokens -= 1
        return True

    async def allow(self, identifier: str) -> bool:
        """Thread-safe guard for token consumption."""

        async with self._lock:
            return self.consume(identifier)


def _hash_token(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"token:{digest}"


def _identify_request(request: Request) -> str:
    auth_header = request.headers.get("authorization", "")
    scheme, _, credentials = auth_header.partition(" ")
    if scheme.lower() == "bearer" and credentials:
        token = credentials.strip()
        metadata = describe_active_token(token)
        session_id = metadata.get("session_id") if metadata else None
        if session_id:
            return f"session:{session_id}"
        return _hash_token(token)

    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
        if client_ip:
            return f"ip:{client_ip}"

    client = request.client
    if client and client.host:
        return f"ip:{client.host}"
    return "ip:unknown"


REFRESH_RATE_LIMITER = TokenBucketRateLimiter(capacity=10, refill_rate_per_second=10 / 60)


class RefreshRateLimitMiddleware(BaseHTTPMiddleware):
    """Apply rate limiting and attempt tracking to ``/auth/refresh``."""

    def __init__(self, app: ASGIApp, limiter: TokenBucketRateLimiter | None = None) -> None:
        super().__init__(app)
        self.limiter = limiter or REFRESH_RATE_LIMITER

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        path = request.url.path.rstrip("/")
        if path != "/auth/refresh":
            return await call_next(request)

        identifier = _identify_request(request)
        allowed = await self.limiter.allow(identifier)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many refresh attempts. Please slow down.",
                    "retry_after": 60,
                },
                headers={"Retry-After": "60"},
            )

        return await call_next(request)


__all__ = [
    "RefreshRateLimitMiddleware",
    "TokenBucketRateLimiter",
    "REFRESH_RATE_LIMITER",
]
