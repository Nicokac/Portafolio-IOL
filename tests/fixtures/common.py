"""Common test utilities shared across the suite."""

from __future__ import annotations

from typing import Any


_SENTINEL = object()


class DummyCtx:
    """A reusable no-op context manager for mocking Streamlit containers."""

    def __init__(self, *, enter_result: Any = _SENTINEL) -> None:
        self._enter_result = enter_result

    def __enter__(self) -> Any:
        if self._enter_result is _SENTINEL:
            return self
        return self._enter_result

    def __exit__(self, *args: Any) -> bool:
        return False

    def button(self, *args: Any, **kwargs: Any) -> bool:
        """Mimic the button API used by some Streamlit contexts."""
        return False


# Convenient shared instance for callers that do not need customization.
dummy_ctx = DummyCtx()

__all__ = ["DummyCtx", "dummy_ctx"]
