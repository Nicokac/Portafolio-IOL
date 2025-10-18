import csv
import logging
import time
from collections import OrderedDict
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path, PurePath
from threading import Lock
from typing import Any, Callable, Dict, Iterable, Tuple

import streamlit as st

logger = logging.getLogger(__name__)


class Cache:
    """Simple in-memory cache and session state handler with thread safety."""

    def __init__(self) -> None:
        """Session state is delegated to Streamlit's native session."""
        pass

    def cache_resource(
        self, func: Callable | None = None, *, maxsize: int | None = None
    ) -> Callable:
        """Cache resources keyed by user session and function arguments.

        Parameters
        ----------
        maxsize:
            Optional maximum number of cached entries. When exceeded, the oldest
            entry is evicted. ``None`` disables the limit.
        """

        def decorator(func: Callable) -> Callable:
            resources: "OrderedDict[Tuple[Callable, Any, Any], Any]" = OrderedDict()
            lock = Lock()

            def _session_key() -> Any:
                return st.session_state.get("session_id")

            # <== De 'main': Función robusta para hashear argumentos complejos.
            def make_hashable(value: Any) -> Any:
                """Convert values into hashable, comparable representations."""
                if isinstance(value, dict):
                    return tuple(
                        (k, make_hashable(v)) for k, v in sorted(value.items())
                    )
                if isinstance(value, (list, tuple)):
                    return tuple(make_hashable(v) for v in value)
                if isinstance(value, (set, frozenset)):
                    return tuple(sorted((make_hashable(v) for v in value), key=repr))
                if isinstance(value, PurePath):
                    return str(value)
                return value

            # <== De 'main': Usa 'make_hashable' para crear una clave segura.
            def _arg_key(args: tuple, kwargs: dict) -> Any:
                ordered_kwargs = tuple(sorted(kwargs.items()))
                return make_hashable((args, ordered_kwargs))

            @wraps(func)
            def wrapper(*args, **kwargs):
                # <== De tu rama: Un chequeo inicial para la nueva funcionalidad.
                if maxsize is not None and maxsize <= 0:
                    return func(*args, **kwargs)

                key = (func, _session_key(), _arg_key(args, kwargs))
                with lock:
                    if key in resources:
                        return resources[key]

                    result = func(*args, **kwargs)
                    resources[key] = result
                    
                    # <== De tu rama: Lógica para limitar el tamaño máximo de la caché.
                    if maxsize is not None:
                        while len(resources) > maxsize:
                            resources.popitem(last=False)
                    return result

            # <== De 'main': La función 'clear' mejorada que acepta argumentos.
            def clear(*call_args, key: Any | None = None, **call_kwargs) -> None:
                sid = _session_key()
                with lock:
                    if key is None and not call_args and not call_kwargs:
                        to_del = [k for k in resources if k[0] is func and k[1] == sid]
                        for k in to_del:
                            resources.pop(k, None)
                    else:
                        if call_args or call_kwargs:
                            key = _arg_key(call_args, call_kwargs)
                        resources.pop((func, sid, key), None)
            
            wrapper.clear = clear
            return wrapper

        if func is not None:
            return decorator(func)
        return decorator

    def cache_data(self, ttl: int | None = None, *, maxsize: int | None = None) -> Callable:
        """Decorator to cache data with optional TTL and maximum size."""

        def decorator(func: Callable) -> Callable:
            cache: "OrderedDict[Tuple[Any, ...], Any]" = OrderedDict()
            timestamps: Dict[Tuple[Any, ...], float] = {}
            lock = Lock()

            def make_hashable(obj: Any) -> Any:
                """Recursively convert unhashable containers to hashable ones."""
                if isinstance(obj, dict):
                    return tuple((k, make_hashable(v)) for k, v in sorted(obj.items()))
                if isinstance(obj, (list, tuple, set)):
                    return tuple(make_hashable(x) for x in obj)
                return obj

            def cleanup(now: float) -> None:
                if ttl is None:
                    return
                expired = [k for k, ts in timestamps.items() if now - ts >= ttl]
                for k in expired:
                    cache.pop(k, None)
                    timestamps.pop(k, None)

            @wraps(func)
            def wrapper(*args, **kwargs):
                if maxsize is not None and maxsize <= 0:
                    return func(*args, **kwargs)

                key = (make_hashable(args), make_hashable(sorted(kwargs.items())))
                now = time.time()
                with lock:
                    cleanup(now)
                    if key in cache and (ttl is None or now - timestamps[key] < ttl):
                        return cache[key]
                result = func(*args, **kwargs)
                with lock:
                    cleanup(time.time())
                    cache[key] = result
                    timestamps[key] = now
                    if maxsize is not None:
                        while len(cache) > maxsize:
                            oldest_key, _ = cache.popitem(last=False)
                            timestamps.pop(oldest_key, None)
                return result

            def clear() -> None:
                with lock:
                    cache.clear()
                    timestamps.clear()

            wrapper.clear = clear
            return wrapper

        return decorator

    # Helper methods for session_state access delegating to Streamlit
    def get(self, key: str, default: Any = None) -> Any:
        return st.session_state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        st.session_state[key] = value

    def pop(self, key: str, default: Any = None) -> Any:
        return st.session_state.pop(key, default)

    def update(self, other: Dict[str, Any]) -> None:
        st.session_state.update(other)

    def clear(self) -> None:
        st.session_state.clear()


# Global cache instance used across the application
cache = Cache()


class VisualCacheRegistry:
    """In-memory index of rendered visuals and their reuse statistics."""

    def __init__(self, *, log_path: str | Path | None = None) -> None:
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._stats: Dict[str, int] = {"hits": 0, "misses": 0}
        self._lock = Lock()
        self._log_path = Path(log_path or "visual_cache_invalidations.csv")

    @staticmethod
    def _normalize_component(component: str) -> str:
        return component.strip().lower() or "unknown"

    @staticmethod
    def _normalize_hash(dataset_hash: str | None) -> str:
        return str(dataset_hash or "none")

    @staticmethod
    def _hash_signature(signature: Any) -> Any:
        if isinstance(signature, (str, bytes, type(None))):
            return signature
        if isinstance(signature, Iterable) and not isinstance(signature, dict):
            try:
                return tuple(VisualCacheRegistry._hash_signature(v) for v in signature)
            except TypeError:
                return str(signature)
        if isinstance(signature, dict):
            try:
                return tuple(
                    (k, VisualCacheRegistry._hash_signature(v))
                    for k, v in sorted(signature.items())
                )
            except Exception:
                return str(signature)
        return signature

    def record(
        self,
        component: str,
        *,
        dataset_hash: str | None,
        reused: bool,
        signature: Any = None,
    ) -> None:
        """Track reuse information for ``component`` bound to ``dataset_hash``."""

        normalized_component = self._normalize_component(component)
        normalized_hash = self._normalize_hash(dataset_hash)
        hashed_signature = self._hash_signature(signature)
        now = time.time()

        with self._lock:
            previous = self._entries.get(normalized_component)
            hits = previous.get("hits", 0) if isinstance(previous, dict) else 0
            misses = previous.get("misses", 0) if isinstance(previous, dict) else 0
            if reused:
                hits += 1
                self._stats["hits"] = self._stats.get("hits", 0) + 1
            else:
                misses += 1
                self._stats["misses"] = self._stats.get("misses", 0) + 1

            self._entries[normalized_component] = {
                "dataset_hash": normalized_hash,
                "signature": hashed_signature,
                "last_used": now,
                "last_status": "hit" if reused else "miss",
                "hits": hits,
                "misses": misses,
            }

    def invalidate(
        self,
        component: str,
        *,
        reason: str,
        dataset_hash: str | None = None,
    ) -> None:
        """Drop the entry for ``component`` and log the invalidation."""

        normalized_component = self._normalize_component(component)
        with self._lock:
            entry = self._entries.pop(normalized_component, None)
        affected_hash = dataset_hash
        if affected_hash is None and isinstance(entry, dict):
            affected_hash = entry.get("dataset_hash")
        self._log_invalidation(reason, affected_hash, normalized_component)

    def invalidate_dataset(self, dataset_hash: str | None, *, reason: str) -> None:
        """Remove every entry that references ``dataset_hash``."""

        normalized_hash = self._normalize_hash(dataset_hash)
        with self._lock:
            affected = [
                component
                for component, entry in self._entries.items()
                if entry.get("dataset_hash") == normalized_hash
            ]
            for component in affected:
                self._entries.pop(component, None)
        for component in affected:
            self._log_invalidation(reason, normalized_hash, component)

    def invalidate_all(self, *, reason: str) -> None:
        """Remove every entry stored in the registry."""

        with self._lock:
            components = list(self._entries.keys())
            entries = {comp: self._entries.pop(comp) for comp in components}
        for component in components:
            entry = entries.get(component) if isinstance(entries, dict) else None
            dataset_hash = None
            if isinstance(entry, dict):
                dataset_hash = entry.get("dataset_hash")
            self._log_invalidation(reason, dataset_hash, component)

    def snapshot(self) -> Dict[str, Any]:
        """Return a read-only snapshot of registry state."""

        with self._lock:
            entries = {
                component: dict(value)
                for component, value in self._entries.items()
                if isinstance(value, dict)
            }
            stats = dict(self._stats)
        return {"entries": entries, "stats": stats}

    def reset(self) -> None:
        """Clear entries and statistics. Intended for tests."""

        with self._lock:
            self._entries.clear()
            self._stats = {"hits": 0, "misses": 0}

    def _log_invalidation(
        self,
        reason: str,
        dataset_hash: str | None,
        component: str,
    ) -> None:
        if not reason:
            return
        normalized_hash = self._normalize_hash(dataset_hash)
        timestamp = datetime.now(UTC).isoformat(timespec="seconds")
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.debug(
                "No se pudo crear el directorio para registrar invalidaciones de caché visual",
                exc_info=True,
            )
        try:
            file_exists = self._log_path.exists()
            with self._log_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if not file_exists:
                    writer.writerow(["timestamp", "reason", "dataset_hash", "component"])
                writer.writerow([timestamp, reason, normalized_hash, component])
        except Exception:
            logger.debug(
                "No se pudo registrar invalidación de caché visual para %s", component, exc_info=True
            )


visual_cache_registry = VisualCacheRegistry()


def cached(func: Callable | None = None, *, ttl: int | None = None, maxsize: int | None = None) -> Callable:
    """Convenience decorator wrapping :meth:`Cache.cache_data`.

    Parameters
    ----------
    func:
        Optional function being decorated when ``@cached`` is used without
        arguments.
    ttl:
        Time-to-live for cached entries in seconds. ``None`` disables
        expiration.
    maxsize:
        Maximum number of cached entries allowed. ``None`` keeps the cache
        unbounded.
    """

    def _decorate(inner: Callable) -> Callable:
        wrapped = cache.cache_data(ttl=ttl, maxsize=maxsize)(inner)
        setattr(wrapped, "cache_ttl", ttl)
        setattr(wrapped, "cache_maxsize", maxsize)
        return wrapped

    if func is not None and callable(func):
        return _decorate(func)
    return _decorate


__all__ = ["Cache", "cache", "cached"]
