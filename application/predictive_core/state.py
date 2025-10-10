"""State containers to track cache usage statistics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class PredictiveCacheState:
    """In-memory counters for cache usage statistics."""

    hits: int = 0
    misses: int = 0
    last_updated: str = "-"
    ttl_hours: float | None = None
    _last_updated_monotonic: float | None = field(default=None, repr=False)

    def record_hit(
        self,
        *,
        last_updated: str | None = None,
        ttl_hours: float | None = None,
    ) -> None:
        """Increase the hit counter and optionally refresh the timestamp."""

        self.hits += 1
        self._last_updated_monotonic = time.monotonic()
        if last_updated:
            self.last_updated = str(last_updated)
        if ttl_hours is not None:
            self.ttl_hours = float(ttl_hours)

    def record_miss(
        self,
        *,
        last_updated: str | None = None,
        ttl_hours: float | None = None,
    ) -> None:
        """Increase the miss counter and optionally refresh the timestamp."""

        self.misses += 1
        self._last_updated_monotonic = time.monotonic()
        if last_updated:
            self.last_updated = str(last_updated)
        if ttl_hours is not None:
            self.ttl_hours = float(ttl_hours)

    def expired(self) -> bool:
        ttl = self.ttl_hours
        if ttl is None:
            return False
        if ttl <= 0:
            return True
        if self._last_updated_monotonic is None:
            return False
        elapsed = time.monotonic() - self._last_updated_monotonic
        return elapsed > ttl * 3600.0


__all__ = ["PredictiveCacheState"]
