"""State containers to track cache usage statistics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictiveCacheState:
    """In-memory counters for cache usage statistics."""

    hits: int = 0
    misses: int = 0
    last_updated: str = "-"

    def record_hit(self, *, last_updated: str | None = None) -> None:
        """Increase the hit counter and optionally refresh the timestamp."""

        self.hits += 1
        if last_updated:
            self.last_updated = str(last_updated)

    def record_miss(self, *, last_updated: str | None = None) -> None:
        """Increase the miss counter and optionally refresh the timestamp."""

        self.misses += 1
        if last_updated:
            self.last_updated = str(last_updated)


__all__ = ["PredictiveCacheState"]
