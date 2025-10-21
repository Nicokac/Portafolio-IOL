"""Router modules for the FastAPI backend."""

from importlib import import_module
from typing import Any

__all__ = ["auth", "engine", "metrics", "predictive", "profile"]


def __getattr__(name: str) -> Any:
    """Lazily import router modules to avoid circular import chains."""

    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
