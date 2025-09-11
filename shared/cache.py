import time
from functools import wraps
from typing import Any, Callable, Dict, Tuple


class Cache:
    """Simple in-memory cache and session state handler.

    Provides decorators similar to Streamlit's ``st.cache_*`` and a plain
    dictionary for session state storage. This abstraction allows the rest of
    the codebase to be decoupled from Streamlit's global objects.
    """

    def __init__(self) -> None:
        self.session_state: Dict[str, Any] = {}

    def cache_resource(self, func: Callable) -> Callable:
        """Cache a resource, instantiating it only once."""
        resource: Any = None
        initialized = False

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal resource, initialized
            if not initialized:
                resource = func(*args, **kwargs)
                initialized = True
            return resource

        def clear() -> None:
            nonlocal resource, initialized
            resource = None
            initialized = False

        wrapper.clear = clear
        return wrapper

    def cache_data(self, ttl: int | None = None) -> Callable:
        """Decorator to cache data with an optional TTL in seconds."""

        def decorator(func: Callable) -> Callable:
            cache: Dict[Tuple[Any, ...], Any] = {}
            timestamps: Dict[Tuple[Any, ...], float] = {}

            def make_hashable(obj: Any) -> Any:
                """Recursively convert unhashable containers to hashable ones."""
                if isinstance(obj, dict):
                    return tuple(
                        (k, make_hashable(v)) for k, v in sorted(obj.items())
                    )
                if isinstance(obj, (list, tuple, set)):
                    return tuple(make_hashable(x) for x in obj)
                return obj

            @wraps(func)
            def wrapper(*args, **kwargs):
                key = (make_hashable(args), make_hashable(sorted(kwargs.items())))
                now = time.time()
                if key in cache and (ttl is None or now - timestamps[key] < ttl):
                    return cache[key]
                result = func(*args, **kwargs)
                cache[key] = result
                timestamps[key] = now
                return result

            def clear() -> None:
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
