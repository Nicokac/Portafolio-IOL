import time
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, Tuple


class Cache:
    """Simple in-memory cache and session state handler with thread safety."""

    def __init__(self) -> None:
        self.session_state: Dict[str, Any] = {}

    def cache_resource(self, func: Callable) -> Callable:
        """Cache a resource, instantiating it only once."""
        resource: Any = None
        initialized = False
        lock = Lock()

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal resource, initialized
            with lock:
                if not initialized:
                    resource = func(*args, **kwargs)
                    initialized = True
                return resource

        def clear() -> None:
            nonlocal resource, initialized
            with lock:
                resource = None
                initialized = False

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

    # Helper methods for session_state access
    def get(self, key: str, default: Any = None) -> Any:
        return self.session_state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.session_state[key] = value

    def pop(self, key: str, default: Any = None) -> Any:
        return self.session_state.pop(key, default)

    def update(self, other: Dict[str, Any]) -> None:
        self.session_state.update(other)


# Global cache instance used across the application
cache = Cache()

__all__ = ["Cache", "cache"]
