"""Base fallback adapter with caching and incident tracking."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import pandas as pd

from services.health import record_adapter_fallback, record_market_data_incident

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdapterProvider:
    """Descriptor for provider callbacks consumed by :class:`BaseMarketDataAdapter`."""

    name: str
    fetcher: Callable[[str, Mapping[str, Any]], pd.DataFrame]


@dataclass
class _CacheEntry:
    data: pd.DataFrame
    provider: str
    timestamp: float


class BaseMarketDataAdapter:
    """Retry helper that orchestrates providers, caching and incidents."""

    cache_ttl: float | None
    incident_source: str

    def __init__(
        self,
        *,
        providers: Sequence[AdapterProvider],
        cache_ttl: float | None = None,
        incident_source: str | None = None,
    ) -> None:
        if not providers:
            raise ValueError("At least one provider must be registered")
        self._providers: tuple[AdapterProvider, ...] = tuple(providers)
        self.cache_ttl = None if cache_ttl is None else max(float(cache_ttl), 0.0)
        self.incident_source = incident_source or self.__class__.__name__
        self._cache: Dict[Any, _CacheEntry] = {}
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch(self, symbol: str, **params: Any) -> pd.DataFrame:
        """Fetch data using the configured providers with fallback semantics."""

        key = self._make_cache_key(symbol, params)
        last_error: Exception | None = None

        for provider in self._providers:
            try:
                payload = provider.fetcher(symbol, params)
                frame = self._normalize_payload(payload, provider.name)
                self._store_cache(key, frame, provider.name)
                record_market_data_incident(
                    adapter=self.incident_source,
                    provider=provider.name,
                    status="success",
                    fallback=bool(last_error),
                )
                record_adapter_fallback(
                    adapter=self.incident_source,
                    provider=provider.name,
                    status="success",
                    fallback=bool(last_error),
                )
                return frame.copy(deep=True)
            except Exception as exc:  # pragma: no cover - defensive log aggregator
                record_adapter_fallback(
                    adapter=self.incident_source,
                    provider=provider.name,
                    status="error",
                    fallback=bool(last_error),
                )
                last_error = exc
                logger.debug(
                    "provider %s failed for %s with params=%s: %s",
                    provider.name,
                    symbol,
                    params,
                    exc,
                )
                record_market_data_incident(
                    adapter=self.incident_source,
                    provider=provider.name,
                    status="error",
                    detail=str(exc),
                )
                continue

        cached = self._get_cached(key)
        if cached is not None:
            record_market_data_incident(
                adapter=self.incident_source,
                provider=cached.provider,
                status="success",
                fallback=True,
                detail="cache-hit",
            )
            return cached.data.copy(deep=True)

        record_market_data_incident(
            adapter=self.incident_source,
            provider="none",
            status="error",
            detail="no-data",
            fallback=True,
        )
        return self.empty_payload()

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------
    def empty_payload(self) -> pd.DataFrame:
        """Return a sentinel payload when no provider and cache succeeded."""

        return pd.DataFrame()

    def _normalize_payload(self, payload: Any, provider: str) -> pd.DataFrame:
        """Validate provider output ensuring a pandas :class:`DataFrame`."""

        if not isinstance(payload, pd.DataFrame):
            raise TypeError(f"Provider {provider} returned unsupported payload {type(payload)!r}")
        if payload.empty:
            raise ValueError(f"Provider {provider} returned an empty payload")
        normalized = payload.copy()
        if not isinstance(normalized.index, pd.DatetimeIndex):
            try:
                normalized.index = pd.to_datetime(normalized.index)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Provider {provider} produced a non temporal index") from exc
        normalized = normalized.sort_index()
        return normalized

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_cache_key(self, symbol: str, params: Mapping[str, Any]) -> Tuple[Any, ...]:
        def _normalize(value: Any) -> Any:
            if isinstance(value, Mapping):
                return tuple((k, _normalize(v)) for k, v in sorted(value.items()))
            if isinstance(value, (list, tuple, set, frozenset)):
                return tuple(_normalize(v) for v in value)
            return value

        ordered = tuple(sorted(params.items()))
        return (symbol.upper(),) + _normalize(ordered)

    def _purge_expired(self) -> None:
        if self.cache_ttl is None:
            return
        if self.cache_ttl <= 0:
            self._cache.clear()
            return
        now = time.time()
        expired = [key for key, entry in self._cache.items() if now - entry.timestamp >= self.cache_ttl]
        for key in expired:
            self._cache.pop(key, None)

    def _store_cache(self, key: Tuple[Any, ...], frame: pd.DataFrame, provider: str) -> None:
        if self.cache_ttl is not None and self.cache_ttl <= 0:
            return
        entry = _CacheEntry(data=frame.copy(deep=True), provider=provider, timestamp=time.time())
        with self._lock:
            self._purge_expired()
            self._cache[key] = entry

    def _get_cached(self, key: Tuple[Any, ...]) -> _CacheEntry | None:
        with self._lock:
            self._purge_expired()
            entry = self._cache.get(key)
            if entry is None:
                return None
            if self.cache_ttl is not None and self.cache_ttl > 0:
                if time.time() - entry.timestamp >= self.cache_ttl:
                    self._cache.pop(key, None)
                    return None
            return entry


__all__ = ["BaseMarketDataAdapter", "AdapterProvider"]
