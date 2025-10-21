"""Lightweight entrypoint for ``python -m controllers`` import checks."""

from . import __build_signature__, __version__

__all__ = ["__version__", "__build_signature__"]

if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    print(f"controllers {__version__} ({__build_signature__})")
