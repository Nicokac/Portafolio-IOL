import time
from functools import wraps
from pathlib import PurePath
from threading import Lock
from typing import Any, Callable, Dict, Tuple

import streamlit as st


class Cache:
    """Simple in-memory cache and session state handler with thread safety."""

    def __init__(self) -> None:
        """Session state is delegated to Streamlit's native session."""
        pass

    def cache_resource(self, func: Callable) -> Callable:
        """Cache resources keyed by user session and function arguments."""

        resources: Dict[Tuple[Callable, Any, Any], Any] = {}
        lock = Lock()

        def _session_key() -> Any:
            return st.session_state.get("session_id")

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

        def _arg_key(args: tuple, kwargs: dict) -> Any:
            ordered_kwargs = tuple(sorted(kwargs.items()))
            return make_hashable((args, ordered_kwargs))

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (func, _session_key(), _arg_key(args, kwargs))
            with lock:
                if key not in resources:
                    resources[key] = func(*args, **kwargs)
                return resources[key]

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

    def cache_data(self, ttl: int | None = None) -> Callable:
        """Decorator to cache data with an optional TTL in seconds."""

        def decorator(func: Callable) -> Callable:
            cache: Dict[Tuple[Any, ...], Any] = {}
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
                key = (make_hashable(args), make_hashable(sorted(kwargs.items())))
                now = time.time()
                with lock:
                    cleanup(now)
                    if key in cache and (ttl is None or now - timestamps[key] < ttl):
                        return cache[key]
                result = func(*args, **kwargs)
                with lock:
                    cache[key] = result
                    timestamps[key] = now
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
