"""Core cache primitives (TTL cache + predictive state)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float | None


class CacheService:
    """Thread-safe TTL cache used for offline fixtures and adapters."""

    def __init__(
        self,
        *,
        namespace: str | None = None,
        monotonic: Callable[[], float] | None = None,
        ttl_override: float | None = None,
    ) -> None:
        self._namespace = (namespace or "").strip()
        self._monotonic = monotonic or time.monotonic
        self._lock = Lock()
        self._store: Dict[str, _CacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._last_updated: datetime | None = None
        self._ttl_override = float(ttl_override) if ttl_override is not None else None
        self._last_ttl: float | None = None
        self._last_expiration: float | None = None

    def _full_key(self, key: str) -> str:
        base_key = str(key)
        return f"{self._namespace}:{base_key}" if self._namespace else base_key

    def _is_expired(self, entry: _CacheEntry) -> bool:
        return entry.expires_at is not None and entry.expires_at <= self._monotonic()

    def get(self, key: str, default: Any = None) -> Any:
        full_key = self._full_key(key)
        with self._lock:
            entry = self._store.get(full_key)
            if entry is None:
                self._misses += 1
                return default
            if self._is_expired(entry):
                self._store.pop(full_key, None)
                self._misses += 1
                if not self._store:
                    self._last_expiration = None
                return default
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> Any:
        full_key = self._full_key(key)
        effective_ttl = self.get_effective_ttl(ttl)
        self._last_ttl = effective_ttl
        if effective_ttl is not None:
            effective_ttl = float(effective_ttl)
            if effective_ttl <= 0:
                with self._lock:
                    self._store.pop(full_key, None)
                    self._last_expiration = None
                return value
            expires_at = self._monotonic() + effective_ttl
        else:
            expires_at = None
        with self._lock:
            self._store[full_key] = _CacheEntry(value=value, expires_at=expires_at)
            self._last_updated = datetime.now(timezone.utc)
            self._last_expiration = expires_at
        return value

    def set_ttl_override(self, ttl_override: float | None) -> None:
        with self._lock:
            self._ttl_override = float(ttl_override) if ttl_override is not None else None

    def get_effective_ttl(self, ttl: float | None = None) -> float | None:
        if ttl is not None:
            ttl = float(ttl)
        if self._ttl_override is not None:
            return self._ttl_override
        if ttl is not None:
            return ttl
        return self._last_ttl

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def get_or_set(
        self,
        key: str,
        loader: Callable[[], Any],
        *,
        ttl: float | None = None,
    ) -> Any:
        sentinel = object()
        cached_value = self.get(key, sentinel)
        if cached_value is not sentinel:
            return cached_value
        value = loader()
        self.set(key, value, ttl=ttl)
        return value

    def invalidate(self, key: str) -> None:
        full_key = self._full_key(key)
        with self._lock:
            self._store.pop(full_key, None)
            if not self._store:
                self._last_expiration = None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._last_updated = None
            self._last_ttl = None
            self._last_expiration = None

    def remaining_ttl(self) -> float | None:
        """Return the remaining time-to-live for the most recent cache entry."""

        with self._lock:
            if not self._store:
                return None
            if self._last_expiration is None:
                return None
            remaining = self._last_expiration - self._monotonic()
            if remaining <= 0:
                return None
            return remaining

    def hit_ratio(self) -> float:
        total = self._hits + self._misses
        if total <= 0:
            return 0.0
        return float(self._hits) / float(total) * 100.0

    @property
    def last_updated(self) -> datetime | None:
        return self._last_updated

    @property
    def last_updated_human(self) -> str:
        if self._last_updated is None:
            return "-"
        return self._last_updated.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def stats(self) -> Dict[str, Any]:
        """Expose cache statistics for reporting and diagnostics."""

        with self._lock:
            return {
                "namespace": self._namespace,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": self.hit_ratio(),
                "last_updated": self.last_updated_human,
                "ttl_seconds": self.get_effective_ttl(),
                "remaining_ttl": self.remaining_ttl(),
            }


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


def _default_stats_payload() -> Dict[str, Any]:
    return {
        "namespace": "",
        "hits": 0,
        "misses": 0,
        "hit_ratio": 0.0,
        "last_updated": "-",
        "ttl_seconds": None,
        "remaining_ttl": None,
    }


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_core_stats(stats: Mapping[str, Any] | None) -> Dict[str, Any]:
    payload = _default_stats_payload()
    if not isinstance(stats, Mapping):
        return payload
    payload.update(
        {
            "namespace": str(stats.get("namespace", "") or ""),
            "hits": int(stats.get("hits", 0) or 0),
            "misses": int(stats.get("misses", 0) or 0),
            "hit_ratio": float(stats.get("hit_ratio", 0.0) or 0.0),
            "last_updated": str(stats.get("last_updated", "-") or "-"),
        }
    )
    ttl_seconds = _safe_float(stats.get("ttl_seconds"))
    remaining_ttl = _safe_float(stats.get("remaining_ttl"))
    payload["ttl_seconds"] = ttl_seconds
    payload["remaining_ttl"] = remaining_ttl
    return payload


def _normalise_snapshot(stats: Mapping[str, Any] | None) -> Dict[str, Any]:
    payload = _default_stats_payload()
    if not isinstance(stats, Mapping):
        return payload
    payload.update(
        {
            "namespace": str(stats.get("namespace", "predictive") or "predictive"),
            "hits": int(stats.get("hits", 0) or 0),
            "misses": int(stats.get("misses", 0) or 0),
            "hit_ratio": float(stats.get("hit_ratio", 0.0) or 0.0),
            "last_updated": str(stats.get("last_updated", "-") or "-"),
        }
    )
    ttl_hours = _safe_float(stats.get("ttl_hours"))
    payload["ttl_seconds"] = ttl_hours * 3600.0 if ttl_hours is not None else None
    payload["remaining_ttl"] = _safe_float(stats.get("remaining_ttl"))
    return payload


def get_cache_stats(cache: CacheService | None = None) -> Dict[str, Any]:
    """Expose cache statistics for a given cache or the predictive engine."""

    if isinstance(cache, CacheService):
        return _normalise_core_stats(cache.stats())
    try:
        from application.predictive_service import (  # pylint: disable=import-outside-toplevel
            PredictiveCacheSnapshot,
        )
        from application.predictive_service import (
            get_cache_stats as _predictive_cache_stats,
        )
    except Exception:  # pragma: no cover - defensive guard
        return _default_stats_payload()
    snapshot = _predictive_cache_stats()
    if snapshot is None:
        return _default_stats_payload()
    if isinstance(snapshot, PredictiveCacheSnapshot):
        return _normalise_snapshot(snapshot.to_dict())
    if isinstance(snapshot, Mapping):
        return _normalise_snapshot(snapshot)
    if is_dataclass(snapshot):
        return _normalise_snapshot(asdict(snapshot))
    stats_dict = getattr(snapshot, "__dict__", None)
    if isinstance(stats_dict, Mapping):
        return _normalise_snapshot(stats_dict)
    return _default_stats_payload()


__all__ = ["CacheService", "PredictiveCacheState", "get_cache_stats"]
