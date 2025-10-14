"""Shared cache helpers for market history and fundamentals."""
from __future__ import annotations

import logging
import os
import pickle
import sqlite3
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from fnmatch import fnmatch
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Sequence

from shared.errors import CacheUnavailableError

import pandas as pd

from services.cache.core import CacheService
from shared.settings import (
    market_data_cache_backend,
    market_data_cache_path,
    market_data_cache_redis_url,
    market_data_cache_ttl,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = float(market_data_cache_ttl)
PREDICTION_TTL_SECONDS = 4 * 60 * 60


def _normalize_symbol(symbol: str | None) -> str:
    if symbol is None:
        return ""
    text = str(symbol).strip()
    return text.upper()


_ALLOWED_SYMBOL_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^=/+&_")


def is_probably_valid_symbol(symbol: str | None) -> bool:
    """Return ``True`` when the ticker contains only supported characters."""

    normalized = _normalize_symbol(symbol)
    if not normalized:
        return False
    return all(char in _ALLOWED_SYMBOL_CHARS for char in normalized)


def _normalize_iterable(values: Iterable[str | None]) -> tuple[str, ...]:
    normalized = [_normalize_symbol(value) for value in values if _normalize_symbol(value)]
    return tuple(sorted(dict.fromkeys(normalized)))


def _strip_namespace(cache: CacheService, full_key: str) -> str:
    namespace = getattr(cache, "_namespace", "")
    if namespace:
        prefix = f"{namespace}:"
        if full_key.startswith(prefix):
            return full_key[len(prefix) :]
    return full_key


def _snapshot_store(cache: CacheService) -> list[tuple[str, Any]]:
    store = getattr(cache, "_store", {})
    lock = getattr(cache, "_lock", None)
    if lock is not None:
        with lock:
            return list(store.items())
    return list(store.items())


def _estimate_size(value: Any) -> int:
    try:
        return len(pickle.dumps(value))
    except Exception:  # pragma: no cover - fallback path
        try:
            return sys.getsizeof(value)
        except Exception:  # pragma: no cover - last resort
            return 0


def _clone_dataframe(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas.DataFrame instance for cached value")
    if df.empty:
        return df.copy()
    return df.copy(deep=True)


@dataclass
class StaleWhileRevalidateResult:
    """Result returned when consulting the SWR cache."""

    value: Any
    is_stale: bool
    was_refreshed: bool
    refresh_scheduled: bool
    duration: float | None = None


@dataclass
class _SWRRecord:
    value: Any
    stored_at: float
    ttl: float
    grace: float


class StaleWhileRevalidateCache:
    """Provide stale-while-revalidate semantics on top of ``CacheService``."""

    def __init__(
        self,
        cache: CacheService,
        *,
        default_ttl: float | None = None,
        grace_ttl: float | None = None,
        max_workers: int | None = None,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._cache = cache
        self._default_ttl = float(default_ttl) if default_ttl is not None else 0.0
        self._grace_ttl = float(grace_ttl) if grace_ttl is not None else self._default_ttl
        self._max_workers = max(int(max_workers or 1), 1)
        self._executor = executor
        self._inflight: dict[str, Future[Any]] = {}
        self._lock = Lock()

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        return self._executor

    @staticmethod
    def _build_record(value: Any, *, ttl: float, grace: float, now: float) -> _SWRRecord:
        ttl = max(float(ttl), 0.0)
        grace = max(float(grace), 0.0)
        return _SWRRecord(value=value, stored_at=float(now), ttl=ttl, grace=grace)

    @staticmethod
    def _coerce_entry(entry: Any) -> _SWRRecord | None:
        if isinstance(entry, _SWRRecord):
            return entry
        if isinstance(entry, dict):
            try:
                return _SWRRecord(
                    value=entry.get("value"),
                    stored_at=float(entry.get("stored_at", 0.0)),
                    ttl=float(entry.get("ttl", 0.0)),
                    grace=float(entry.get("grace", 0.0)),
                )
            except (TypeError, ValueError):
                return None
        return None

    def _store_entry(self, key: str, record: _SWRRecord) -> None:
        ttl_total = record.ttl + record.grace
        ttl_total = max(ttl_total, 0.0)
        payload = {
            "value": record.value,
            "stored_at": record.stored_at,
            "ttl": record.ttl,
            "grace": record.grace,
        }
        ttl_arg: float | None = ttl_total if ttl_total > 0 else None
        self._cache.set(key, payload, ttl=ttl_arg)

    def _run_loader(
        self,
        key: str,
        loader: Callable[[], Any],
        *,
        ttl: float,
        grace: float,
        on_refresh: Callable[[float, Any, bool], None] | None,
        background: bool,
    ) -> tuple[Any, float]:
        start = time.perf_counter()
        value = loader()
        duration = float(time.perf_counter() - start)
        record = self._build_record(value, ttl=ttl, grace=grace, now=time.time())
        self._store_entry(key, record)
        if on_refresh is not None:
            try:
                on_refresh(duration, value, background)
            except Exception:  # pragma: no cover - defensive observer errors
                LOGGER.exception("on_refresh callback for %s failed", key)
        return value, duration

    def _schedule_refresh(
        self,
        key: str,
        loader: Callable[[], Any],
        *,
        ttl: float,
        grace: float,
        on_refresh: Callable[[float, Any, bool], None] | None,
    ) -> bool:
        with self._lock:
            if key in self._inflight:
                return False

            executor = self._get_executor()

            def _task() -> None:
                try:
                    self._run_loader(
                        key,
                        loader,
                        ttl=ttl,
                        grace=grace,
                        on_refresh=on_refresh,
                        background=True,
                    )
                except Exception:  # pragma: no cover - loader failures depend on runtime
                    LOGGER.exception("SWR refresh failed for %s", key)
                finally:
                    with self._lock:
                        self._inflight.pop(key, None)

            future = executor.submit(_task)
            self._inflight[key] = future
        return True

    def get_or_refresh(
        self,
        key: str,
        loader: Callable[[], Any],
        *,
        ttl: float | None = None,
        grace: float | None = None,
        now: float | None = None,
        on_refresh: Callable[[float, Any, bool], None] | None = None,
    ) -> StaleWhileRevalidateResult:
        ttl_value = self._default_ttl if ttl is None else float(ttl)
        grace_value = self._grace_ttl if grace is None else float(grace)
        reference = float(time.time() if now is None else now)
        cached = self._cache.get(key)
        record = self._coerce_entry(cached)

        if record is None:
            value, duration = self._run_loader(
                key,
                loader,
                ttl=ttl_value,
                grace=grace_value,
                on_refresh=on_refresh,
                background=False,
            )
            return StaleWhileRevalidateResult(
                value=value,
                is_stale=False,
                was_refreshed=True,
                refresh_scheduled=False,
                duration=duration,
            )

        age = max(0.0, reference - record.stored_at)
        if age < record.ttl:
            return StaleWhileRevalidateResult(
                value=record.value,
                is_stale=False,
                was_refreshed=False,
                refresh_scheduled=False,
            )

        expiration = record.ttl + record.grace
        if expiration <= 0 or age >= expiration:
            value, duration = self._run_loader(
                key,
                loader,
                ttl=ttl_value,
                grace=grace_value,
                on_refresh=on_refresh,
                background=False,
            )
            return StaleWhileRevalidateResult(
                value=value,
                is_stale=False,
                was_refreshed=True,
                refresh_scheduled=False,
                duration=duration,
            )

        self._schedule_refresh(
            key,
            loader,
            ttl=ttl_value,
            grace=grace_value,
            on_refresh=on_refresh,
        )
        return StaleWhileRevalidateResult(
            value=record.value,
            is_stale=True,
            was_refreshed=False,
            refresh_scheduled=True,
        )

    def wait(self, timeout: float | None = None) -> None:
        """Block until in-flight refresh operations complete."""

        futures: list[Future[Any]]
        with self._lock:
            futures = list(self._inflight.values())
        for future in futures:
            try:
                future.result(timeout=timeout)
            except Exception:  # pragma: no cover - best effort draining
                pass

    def shutdown(self, wait: bool = True) -> None:
        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=wait)
        self._executor = None


class _BasePersistentBackend:
    """Interface for persistent cache backends."""

    def get(self, key: str) -> tuple[float | None, Any] | None:  # pragma: no cover - interface
        raise NotImplementedError

    def set(self, key: str, value: Any, expires_at: float | None) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def invalidate(self, key: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def clear_namespace(self, namespace: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class _SQLiteBackend(_BasePersistentBackend):
    """Persist cached payloads to a SQLite database."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = Lock()
        self._initialised = False

    def database_path(self) -> Path:
        """Return the on-disk path for the SQLite cache database."""

        return self._path

    def _ensure(self) -> None:
        if self._initialised:
            return
        with self._lock:
            if self._initialised:
                return
            if not self._path.parent.exists():
                self._path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self._path) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, expires_at REAL, payload BLOB)"
                )
                conn.commit()
            self._initialised = True

    def _connection(self) -> sqlite3.Connection:
        self._ensure()
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _safe_stat(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def run_maintenance(
        self, *, vacuum: bool = True, now: float | None = None
    ) -> dict[str, float | int]:
        """Clean expired rows and optionally compact the SQLite cache."""

        self._ensure()
        reference_time = float(now if now is not None else time.time())
        size_before = float(self._safe_stat(self._path))
        deleted = 0
        vacuum_duration = 0.0

        with self._connection() as conn:
            before_changes = conn.total_changes
            conn.execute(
                "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (reference_time,),
            )
            conn.commit()
            delta = conn.total_changes - before_changes
            if delta > 0:
                deleted = int(delta)
            if vacuum:
                start = time.perf_counter()
                conn.execute("VACUUM")
                vacuum_duration = float(time.perf_counter() - start)

        size_after = float(self._safe_stat(self._path))
        return {
            "deleted": deleted,
            "size_before": size_before,
            "size_after": size_after,
            "vacuum_duration": vacuum_duration,
        }

    def get(self, key: str) -> tuple[float | None, Any] | None:
        self._ensure()
        with self._connection() as conn:
            row = conn.execute(
                "SELECT expires_at, payload FROM cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        expires_at, payload = row
        try:
            value = pickle.loads(payload)
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("No se pudo deserializar la clave %s desde SQLite", key)
            return None
        return expires_at, value

    def set(self, key: str, value: Any, expires_at: float | None) -> None:
        self._ensure()
        payload = pickle.dumps(value)
        with self._connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache(key, expires_at, payload) VALUES(?, ?, ?)",
                (key, expires_at, payload),
            )
            conn.commit()

    def invalidate(self, key: str) -> None:
        if not self._initialised:
            return
        with self._connection() as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    def clear_namespace(self, namespace: str) -> None:
        if not self._initialised:
            return
        pattern = f"{namespace}:%%" if namespace else "%"
        with self._connection() as conn:
            conn.execute("DELETE FROM cache WHERE key LIKE ?", (pattern,))
            conn.commit()


class _RedisBackend(_BasePersistentBackend):
    """Persist cache payloads to Redis when available."""

    def __init__(self, url: str) -> None:
        try:
            import redis  # type: ignore import
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Redis backend solicitado pero el paquete no está instalado") from exc

        self._client = redis.Redis.from_url(url)

    def get(self, key: str) -> tuple[float | None, Any] | None:
        payload = self._client.get(key)
        if payload is None:
            return None
        expires_at = self._client.ttl(key)
        absolute_expiration: float | None
        if expires_at is None or expires_at < 0:
            absolute_expiration = None
        else:
            absolute_expiration = time.time() + float(expires_at)
        return absolute_expiration, pickle.loads(payload)

    def set(self, key: str, value: Any, expires_at: float | None) -> None:
        payload = pickle.dumps(value)
        if expires_at is None:
            self._client.set(key, payload)
        else:
            ttl_seconds = max(int(expires_at - time.time()), 1)
            self._client.setex(key, ttl_seconds, payload)

    def invalidate(self, key: str) -> None:
        self._client.delete(key)

    def clear_namespace(self, namespace: str) -> None:
        pattern = f"{namespace}:*" if namespace else "*"
        keys = list(self._client.scan_iter(pattern))
        if keys:
            self._client.delete(*keys)


_BACKEND: _BasePersistentBackend | None = None
_BACKEND_LOCK = Lock()


def _initialise_backend() -> _BasePersistentBackend | None:
    backend_name = (market_data_cache_backend or "").strip().lower() or "memory"
    if backend_name == "memory":
        LOGGER.info("Market data cache configurado en modo memoria (sin persistencia)")
        return None
    if backend_name == "sqlite":
        path_value = market_data_cache_path or "data/market_cache.db"
        path = Path(os.fspath(path_value))
        LOGGER.info("Market data cache persistente inicializado en SQLite: %s", path)
        return _SQLiteBackend(path)
    if backend_name == "redis":
        url = market_data_cache_redis_url
        if not url:
            LOGGER.warning(
                "MARKET_DATA_CACHE_REDIS_URL no definido; usando caché en memoria"
            )
            return None
        LOGGER.info("Market data cache persistente configurado con Redis (%s)", url)
        try:
            return _RedisBackend(url)
        except RuntimeError:
            LOGGER.exception("Fallo al inicializar Redis; se usará caché en memoria")
            return None
    LOGGER.warning(
        "Backend de caché desconocido '%s'; se usará caché en memoria", backend_name
    )
    return None


def _get_backend() -> _BasePersistentBackend | None:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is None:
            _BACKEND = _initialise_backend()
    return _BACKEND


def _extract_symbols_from_key(key: str) -> str | None:
    parts = key.split("|")
    if not parts:
        return None
    label = parts[0]
    if label == "history" and len(parts) >= 4:
        return parts[3] or None
    if label == "fundamentals" and len(parts) >= 2:
        return parts[1] or None
    if label == "predictions" and len(parts) >= 4:
        return parts[3] or None
    return None


class PersistentCacheService(CacheService):
    """CacheService que replica entradas en un backend persistente."""

    def __init__(
        self,
        *,
        namespace: str | None = None,
        monotonic: Callable[[], float] | None = None,
        ttl_override: float | None = None,
        backend: _BasePersistentBackend | None = None,
    ) -> None:
        super().__init__(namespace=namespace, monotonic=monotonic, ttl_override=ttl_override)
        self._persistent_backend = backend

    def _log_event(self, action: str, key: str) -> None:
        symbols = _extract_symbols_from_key(key)
        context = f"symbols={symbols}" if symbols else f"key={key}"
        LOGGER.info(
            "market-data-cache %s %s (%s)",
            self._namespace or "default",
            action,
            context,
        )

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        sentinel = object()
        value = CacheService.get(self, key, sentinel)
        if value is not sentinel:
            self._log_event("hit:memory", key)
            return value
        backend = self._persistent_backend
        if backend is None:
            self._log_event("miss:memory", key)
            return default
        record = backend.get(self._full_key(key))
        if record is None:
            self._log_event("miss:persistent", key)
            return default
        expires_at, payload = record
        now = time.time()
        if expires_at is not None and expires_at <= now:
            backend.invalidate(self._full_key(key))
            self._log_event("miss:expired", key)
            return default
        remaining = None
        if expires_at is not None:
            remaining = max(0.0, expires_at - now)
        CacheService.set(self, key, payload, ttl=remaining)
        self._log_event("hit:persistent", key)
        return payload

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> Any:  # type: ignore[override]
        backend = self._persistent_backend
        effective_ttl = self.get_effective_ttl(ttl)
        result = CacheService.set(self, key, value, ttl=ttl)
        if backend is not None:
            expires_at = None
            if effective_ttl is not None:
                expires_at = time.time() + float(effective_ttl)
            backend.set(self._full_key(key), value, expires_at)
            self._log_event("store", key)
        return result

    def invalidate(self, key: str) -> None:  # type: ignore[override]
        CacheService.invalidate(self, key)
        backend = self._persistent_backend
        if backend is not None:
            backend.invalidate(self._full_key(key))
            self._log_event("invalidate", key)

    def clear(self) -> None:  # type: ignore[override]
        CacheService.clear(self)
        backend = self._persistent_backend
        if backend is not None:
            try:
                backend.clear_namespace(self._namespace or "")
                self._log_event("clear", "*")
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("No se pudo limpiar la caché persistente %s", self._namespace)


def _create_cache(namespace: str) -> PersistentCacheService:
    backend = _get_backend()
    return PersistentCacheService(namespace=namespace, backend=backend)


_GLOBAL_CACHE: MarketDataCache | None = None
_GLOBAL_CACHE_LOCK = Lock()


def get_market_data_cache() -> MarketDataCache:
    """Return a lazily initialised market data cache with persistent backend."""

    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is not None:
        return _GLOBAL_CACHE
    with _GLOBAL_CACHE_LOCK:
        if _GLOBAL_CACHE is None:
            _GLOBAL_CACHE = MarketDataCache()
    return _GLOBAL_CACHE


def create_persistent_cache(namespace: str) -> PersistentCacheService:
    """Expose a helper for services requiring a persistent cache."""

    return _create_cache(namespace)


def run_persistent_cache_maintenance(
    *, now: float | None = None, vacuum: bool = True
) -> dict[str, float | int] | None:
    """Run maintenance tasks for the SQLite persistent cache backend."""

    backend = _get_backend()
    if not isinstance(backend, _SQLiteBackend):
        if backend is None:
            LOGGER.debug(
                "Omitiendo mantenimiento de caché persistente: backend sin inicializar"
            )
        else:
            LOGGER.debug(
                "Omitiendo mantenimiento de caché persistente: backend %s no es SQLite",
                type(backend).__name__,
            )
        return None
    return backend.run_maintenance(now=now, vacuum=vacuum)


def get_sqlite_backend_path() -> Path | None:
    """Return the SQLite path when the persistent cache uses it."""

    backend = _get_backend()
    if isinstance(backend, _SQLiteBackend):
        return backend.database_path()
    return None


@dataclass
class MarketDataCache:
    """Manage cached market datasets with TTL invalidation."""

    history_cache: CacheService
    fundamentals_cache: CacheService
    prediction_cache: CacheService
    default_ttl: float = DEFAULT_TTL_SECONDS
    prediction_ttl: float = PREDICTION_TTL_SECONDS

    def __init__(
        self,
        *,
        history_cache: CacheService | None = None,
        fundamentals_cache: CacheService | None = None,
        prediction_cache: CacheService | None = None,
        default_ttl: float | None = None,
        prediction_ttl: float | None = None,
    ) -> None:
        self.history_cache = history_cache or _create_cache("market_history")
        self.fundamentals_cache = fundamentals_cache or _create_cache(
            "market_fundamentals"
        )
        self.prediction_cache = prediction_cache or _create_cache(
            "market_predictions"
        )
        if default_ttl is not None:
            self.default_ttl = float(default_ttl)
        if prediction_ttl is not None:
            self.prediction_ttl = float(prediction_ttl)

    def _effective_ttl(self, ttl_seconds: float | None) -> float | None:
        if ttl_seconds is None:
            return self.default_ttl
        return float(ttl_seconds)

    def _effective_prediction_ttl(self, ttl_seconds: float | None) -> float | None:
        if ttl_seconds is None:
            return self.prediction_ttl
        return float(ttl_seconds)

    def resolve_prediction_ttl(self, ttl_hours: float | None = None) -> float:
        seconds = None if ttl_hours is None else max(float(ttl_hours), 0.0) * 3600.0
        effective = self._effective_prediction_ttl(seconds)
        if effective is None:
            return 0.0
        return float(effective)

    def _caches(self) -> dict[str, CacheService]:
        caches = {
            "history": self.history_cache,
            "fundamentals": self.fundamentals_cache,
            "predictions": self.prediction_cache,
        }
        for name, cache in caches.items():
            if cache is None:
                raise CacheUnavailableError(f"Cache '{name}' no está inicializada")
        return caches  # type: ignore[return-value]

    def get_status_summary(self) -> dict[str, Any]:
        caches = self._caches()
        total_entries = 0
        total_hits = 0
        total_misses = 0
        ttl_accumulator = 0.0
        ttl_count = 0
        size_bytes = 0

        for cache in caches.values():
            total_hits += getattr(cache, "hits", 0)
            total_misses += getattr(cache, "misses", 0)
            items = _snapshot_store(cache)
            total_entries += len(items)
            monotonic = getattr(cache, "_monotonic", time.monotonic)
            now = float(monotonic())
            for full_key, entry in items:
                base_key = _strip_namespace(cache, full_key)
                value = getattr(entry, "value", None)
                size_bytes += _estimate_size(base_key)
                size_bytes += _estimate_size(value)
                expires_at = getattr(entry, "expires_at", None)
                if expires_at is None:
                    continue
                remaining = float(expires_at) - now
                if remaining < 0:
                    remaining = 0.0
                ttl_accumulator += remaining
                ttl_count += 1

        hit_ratio = 0.0
        if (total_hits + total_misses) > 0:
            hit_ratio = float(total_hits) / float(total_hits + total_misses) * 100.0
        avg_ttl = ttl_accumulator / ttl_count if ttl_count > 0 else None
        size_mb = size_bytes / (1024.0 * 1024.0)

        return {
            "total_entries": total_entries,
            "hit_ratio": hit_ratio,
            "avg_ttl_seconds": avg_ttl,
            "size_mb": size_mb,
        }

    def invalidate_matching(
        self,
        *,
        keys: Sequence[str] | None = None,
        pattern: str | None = None,
    ) -> int:
        caches = self._caches()
        if not keys and not pattern:
            raise ValueError("Se requiere 'pattern' o 'keys' para invalidar entradas")

        unique_keys: set[tuple[str, str]] = set()
        for cache_name, cache in caches.items():
            items = _snapshot_store(cache)
            namespace = getattr(cache, "_namespace", "")
            prefix = f"{namespace}:" if namespace else ""
            for full_key, _entry in items:
                base_key = full_key[len(prefix) :] if prefix and full_key.startswith(prefix) else full_key
                match = False
                if keys and base_key in keys:
                    match = True
                if pattern and fnmatch(base_key, pattern):
                    match = True
                if match:
                    unique_keys.add((cache_name, base_key))

        removed = 0
        for cache_name, base_key in unique_keys:
            cache = caches[cache_name]
            cache.invalidate(base_key)
            removed += 1
        return removed

    def cleanup_expired(self) -> dict[str, int]:
        caches = self._caches()
        expired = 0
        orphans = 0

        for cache in caches.values():
            items = _snapshot_store(cache)
            monotonic = getattr(cache, "_monotonic", time.monotonic)
            now = float(monotonic())
            namespace = getattr(cache, "_namespace", "")
            prefix = f"{namespace}:" if namespace else ""

            for full_key, entry in items:
                base_key = full_key[len(prefix) :] if prefix and full_key.startswith(prefix) else full_key
                expires_at = getattr(entry, "expires_at", None)
                has_value = hasattr(entry, "value")
                if not has_value:
                    cache.invalidate(base_key)
                    orphans += 1
                    continue
                if expires_at is None:
                    continue
                if float(expires_at) <= now:
                    cache.invalidate(base_key)
                    expired += 1

        return {"expired_removed": expired, "orphans_removed": orphans}

    def _history_key(
        self,
        symbols: Sequence[str],
        *,
        period: str,
        benchmark: str | None = None,
        extra: Sequence[str] | None = None,
    ) -> str:
        normalized_symbols = _normalize_iterable(symbols)
        normalized_extra = _normalize_iterable(extra or [])
        benchmark_key = _normalize_symbol(benchmark)
        period_key = str(period or "").strip().lower()
        return "|".join(
            [
                "history",
                period_key or "-",
                benchmark_key or "-",
                ",".join(normalized_symbols) or "-",
                ",".join(normalized_extra) or "-",
            ]
        )

    def _fundamentals_key(
        self,
        symbols: Sequence[str],
        *,
        sectors: Sequence[str] | None = None,
    ) -> str:
        normalized_symbols = _normalize_iterable(symbols)
        sector_key = _normalize_iterable(sectors or [])
        return "|".join(
            [
                "fundamentals",
                ",".join(normalized_symbols) or "-",
                ",".join(sector_key) or "-",
            ]
        )

    def build_prediction_key(
        self,
        symbols: Sequence[str],
        *,
        span: int,
        sectors: Sequence[str] | None = None,
        period: str | None = None,
    ) -> str:
        normalized_symbols = _normalize_iterable(symbols)
        normalized_sectors = _normalize_iterable(sectors or [])
        span_key = str(int(span) if span is not None else 0)
        period_key = str(period or "").strip().lower() or "-"
        return "|".join(
            [
                "predictions",
                span_key,
                period_key,
                ",".join(normalized_symbols) or "-",
                ",".join(normalized_sectors) or "-",
            ]
        )

    def get_history(
        self,
        symbols: Sequence[str],
        *,
        loader: Callable[[], pd.DataFrame],
        period: str,
        benchmark: str | None = None,
        extra: Sequence[str] | None = None,
        ttl_seconds: float | None = None,
    ) -> pd.DataFrame:
        key = self._history_key(symbols, period=period, benchmark=benchmark, extra=extra)
        cached = self.history_cache.get(key)
        if isinstance(cached, pd.DataFrame):
            return _clone_dataframe(cached)  # type: ignore[return-value]
        value = loader()
        cloned = _clone_dataframe(value)
        if cloned is not None:
            self.history_cache.set(key, cloned, ttl=self._effective_ttl(ttl_seconds))
        return value

    def invalidate_history(
        self,
        symbols: Sequence[str],
        *,
        period: str,
        benchmark: str | None = None,
        extra: Sequence[str] | None = None,
    ) -> None:
        key = self._history_key(symbols, period=period, benchmark=benchmark, extra=extra)
        self.history_cache.invalidate(key)

    def get_fundamentals(
        self,
        symbols: Sequence[str],
        *,
        loader: Callable[[], pd.DataFrame],
        sectors: Sequence[str] | None = None,
        ttl_seconds: float | None = None,
    ) -> pd.DataFrame:
        key = self._fundamentals_key(symbols, sectors=sectors)
        cached = self.fundamentals_cache.get(key)
        if isinstance(cached, pd.DataFrame):
            return _clone_dataframe(cached)  # type: ignore[return-value]
        value = loader()
        cloned = _clone_dataframe(value)
        if cloned is not None:
            self.fundamentals_cache.set(key, cloned, ttl=self._effective_ttl(ttl_seconds))
        return value

    def invalidate_fundamentals(
        self,
        symbols: Sequence[str],
        *,
        sectors: Sequence[str] | None = None,
    ) -> None:
        key = self._fundamentals_key(symbols, sectors=sectors)
        self.fundamentals_cache.invalidate(key)

    def get_predictions(
        self,
        symbols: Sequence[str],
        *,
        loader: Callable[[], pd.DataFrame],
        span: int,
        sectors: Sequence[str] | None = None,
        period: str | None = None,
        ttl_seconds: float | None = None,
    ) -> tuple[pd.DataFrame, bool]:
        key = self.build_prediction_key(
            symbols,
            span=span,
            sectors=sectors,
            period=period,
        )
        cached = self.prediction_cache.get(key)
        if isinstance(cached, pd.DataFrame):
            return _clone_dataframe(cached) or pd.DataFrame(), True  # type: ignore[return-value]
        value = loader()
        cloned = _clone_dataframe(value)
        if cloned is not None:
            self.prediction_cache.set(
                key,
                cloned,
                ttl=self._effective_prediction_ttl(ttl_seconds),
            )
        return value, False

    def invalidate_predictions(
        self,
        symbols: Sequence[str],
        *,
        span: int,
        sectors: Sequence[str] | None = None,
        period: str | None = None,
    ) -> None:
        key = self.build_prediction_key(
            symbols,
            span=span,
            sectors=sectors,
            period=period,
        )
        self.prediction_cache.invalidate(key)


__all__ = [
    "MarketDataCache",
    "get_market_data_cache",
    "create_persistent_cache",
    "DEFAULT_TTL_SECONDS",
    "run_persistent_cache_maintenance",
    "get_sqlite_backend_path",
    "StaleWhileRevalidateCache",
    "StaleWhileRevalidateResult",
    "is_probably_valid_symbol",
]
