"""Regression tests covering Streamlit fragment compatibility."""

from __future__ import annotations

import importlib.util
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_lazy_runtime():
    runtime_path = Path(__file__).resolve().parents[2] / "ui" / "lazy" / "runtime.py"
    module_name = "ui.lazy.runtime_test"
    spec = importlib.util.spec_from_file_location(module_name, runtime_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _DecoratorFragmentStreamlit:
    def __init__(self) -> None:
        self.container_enters = 0

    def fragment(self):
        def _decorator(func):
            def _wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return _wrapper

        return _decorator

    def container(self):
        @contextmanager
        def _ctx():
            self.container_enters += 1
            yield "container"

        return _ctx()


class _NoFragmentStreamlit:
    def __init__(self) -> None:
        self.container_enters = 0

    def container(self):
        @contextmanager
        def _ctx():
            self.container_enters += 1
            yield "container"

        return _ctx()


def test_lazy_fragment_handles_decorator_api(monkeypatch: pytest.MonkeyPatch) -> None:
    lazy_runtime = _load_lazy_runtime()
    fake_st = _DecoratorFragmentStreamlit()
    monkeypatch.setattr(lazy_runtime, "st", fake_st)
    monkeypatch.setattr(lazy_runtime, "_FRAGMENT_WARNING_EMITTED", False)
    guardian = SimpleNamespace(wait_for_hydration=lambda *_, **__: True)
    monkeypatch.setattr(lazy_runtime, "get_fragment_state_guardian", lambda: guardian)

    with lazy_runtime.lazy_fragment("portfolio_table", component="table") as fragment_ctx:
        assert fragment_ctx.scope == "fragment"

    assert fake_st.container_enters == 1


def test_lazy_fragment_fallback_without_fragment_api(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    lazy_runtime = _load_lazy_runtime()
    fake_st = _NoFragmentStreamlit()
    monkeypatch.setattr(lazy_runtime, "st", fake_st)
    monkeypatch.setattr(lazy_runtime, "_FRAGMENT_WARNING_EMITTED", False)
    guardian = SimpleNamespace(wait_for_hydration=lambda *_, **__: True)
    monkeypatch.setattr(lazy_runtime, "get_fragment_state_guardian", lambda: guardian)

    caplog.set_level(logging.WARNING)

    with lazy_runtime.lazy_fragment("portfolio_table", component="table") as fragment_ctx:
        assert fragment_ctx.scope == "global"

    assert fake_st.container_enters == 1
    assert "fallback to container" in caplog.text
