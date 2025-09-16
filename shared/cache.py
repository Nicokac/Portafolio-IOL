import time
from collections import OrderedDict
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, Tuple

import streamlit as st


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

            def _arg_key(args: tuple, kwargs: dict) -> Any:
                if args:
                    return args[0]
                return (args, tuple(sorted(kwargs.items())))

            @wraps(func)
            def wrapper(*args, **kwargs):
                if maxsize is not None and maxsize <= 0:
                    return func(*args, **kwargs)

                key = (func, _session_key(), _arg_key(args, kwargs))
                with lock:
                    if key in resources:
                        return resources[key]

                    result = func(*args, **kwargs)
                    resources[key] = result
                    if maxsize is not None:
                        while len(resources) > maxsize:
                            resources.popitem(last=False)
                    return result

            def clear(key: Any | None = None) -> None:
                sid = _session_key()
                with lock:
                    if key is None:
                        to_del = [k for k in resources if k[0] is func and k[1] == sid]
                        for k in to_del:
                            resources.pop(k, None)
                    else:
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

__all__ = ["Cache", "cache"]
