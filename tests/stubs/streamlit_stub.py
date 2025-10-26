"""Minimal Streamlit stub used during test collection and execution."""

from __future__ import annotations

import contextlib
import types
from typing import Any, Iterable


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


class _NoOpContext(contextlib.AbstractContextManager):
    """Context manager that ignores all operations."""

    def __enter__(self) -> "_NoOpContext":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


def _context_factory(*_args: Any, **_kwargs: Any) -> _NoOpContext:
    return _NoOpContext()


def _cache_data(*decorator_args: Any, **decorator_kwargs: Any):
    if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1 and not decorator_kwargs:
        return decorator_args[0]

    def _decorator(func):
        return func

    return _decorator


def _tabs(labels: Iterable[Any]) -> list[_NoOpContext]:
    return [_NoOpContext() for _ in labels]


class _Namespace(types.SimpleNamespace):
    def __call__(self, *_args: Any, **_kwargs: Any) -> _NoOpContext:
        return _NoOpContext()

    def __getattr__(self, _name: str):  # pragma: no cover - defensive
        return _noop

    def container(self, *_args: Any, **_kwargs: Any) -> _NoOpContext:
        return _NoOpContext()

    def tabs(self, labels: Iterable[Any]) -> list[_NoOpContext]:
        return _tabs(labels)


st = types.ModuleType("streamlit")
st.session_state = {}
st.cache_data = _cache_data
st.cache_resource = _cache_data
st.form = _context_factory
st.sidebar = _Namespace()
st.warning = _noop
st.info = _noop
st.write = _noop
st.markdown = _noop
st.toast = _noop
st.empty = _context_factory
st.container = _context_factory
st.tabs = _tabs
st.experimental_rerun = _noop
st.experimental_set_query_params = _noop
st.secrets = {}
st.stop = _noop


def __getattr__(name: str):  # pragma: no cover - fallback for unexpected usage
    return _noop


setattr(st, "__getattr__", __getattr__)
