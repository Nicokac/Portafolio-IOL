"""Regression tests for the lazy fragment Streamlit factory."""

from __future__ import annotations

from contextlib import contextmanager

import logging

import pytest

import ui.lazy.runtime as lazy_runtime


class _FunctionFragmentStreamlit:
    def __init__(self) -> None:
        self.fragment_calls: list[str] = []

    def fragment(self, name: str):
        self.fragment_calls.append(name)

        def _builder():
            @contextmanager
            def _ctx():
                yield name

            return _ctx()

        return _builder


class _ContainerStreamlit:
    def __init__(self) -> None:
        self.entered = 0

    def container(self):
        @contextmanager
        def _ctx():
            self.entered += 1
            yield "container"

        return _ctx()


def test_fragment_factory_wraps_function_return(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = _FunctionFragmentStreamlit()
    monkeypatch.setattr(lazy_runtime, "st", fake_st)
    monkeypatch.setattr(lazy_runtime, "_FRAGMENT_WARNING_EMITTED", False)

    factory = lazy_runtime._fragment_factory()
    assert callable(factory)

    with factory("portfolio_table") as context_name:
        assert context_name == "portfolio_table"

    assert fake_st.fragment_calls == ["portfolio_table"]


def test_fragment_factory_fallback_to_container(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    fake_st = _ContainerStreamlit()
    monkeypatch.setattr(lazy_runtime, "st", fake_st)
    monkeypatch.setattr(lazy_runtime, "_FRAGMENT_WARNING_EMITTED", False)

    caplog.set_level(logging.WARNING)

    factory = lazy_runtime._fragment_factory()
    assert factory is None
    assert "fallback to container" in caplog.text

    with lazy_runtime._enter_scope(
        "portfolio_table", fragment_factory=None, form_callable=None, scope="global"
    ):
        pass

    assert fake_st.entered == 1
