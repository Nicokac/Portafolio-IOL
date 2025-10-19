"""Regression tests for lazy fragment hydration waits."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import ui.lazy.runtime as lazy_runtime


class _FragmentStreamlitStub:
    def __init__(self) -> None:
        self.fragment_calls: list[str] = []
        self.experimental_rerun = MagicMock()
        self.session_state: dict[str, object] = {}

    def fragment(self, name: str | None = None):  # noqa: D401 - mimic Streamlit signature
        @contextmanager
        def _ctx():
            self.fragment_calls.append(str(name or ""))
            yield SimpleNamespace()

        return _ctx()

    def container(self):  # pragma: no cover - compatibility helper
        @contextmanager
        def _ctx():
            yield SimpleNamespace()

        return _ctx()

    def empty(self):  # pragma: no cover - compatibility helper
        placeholder = SimpleNamespace()
        placeholder.container = self.container  # type: ignore[attr-defined]
        return placeholder


class _FakeTime:
    def __init__(self) -> None:
        self._now = 0.0

    def perf_counter(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now += max(float(seconds), 0.0)
        lazy_runtime.fragment_context_ready = True

    def time(self) -> float:  # pragma: no cover - compatibility helper
        return self._now


def test_lazy_fragment_wait_avoids_fallback(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    telemetry_events: list[dict[str, object]] = []

    def _capture_telemetry(**kwargs):
        telemetry_events.append(kwargs)

    fake_st = _FragmentStreamlitStub()
    fake_time = _FakeTime()
    guardian = SimpleNamespace(wait_for_hydration=lambda *_, **__: True)

    monkeypatch.setattr(lazy_runtime, "st", fake_st)
    monkeypatch.setattr(lazy_runtime, "time", fake_time)
    monkeypatch.setattr(lazy_runtime, "_FRAGMENT_WARNING_EMITTED", False)
    monkeypatch.setattr(lazy_runtime, "_FRAGMENT_CONTEXT_RERUN_DATASETS", set())
    monkeypatch.setattr(lazy_runtime, "get_fragment_state_guardian", lambda: guardian)
    monkeypatch.setattr(lazy_runtime, "log_default_telemetry", _capture_telemetry, raising=False)

    lazy_runtime.fragment_context_ready = False

    caplog.set_level(logging.INFO)

    with lazy_runtime.lazy_fragment("portfolio_table", component="table", dataset_token="abc") as fragment_ctx:
        assert fragment_ctx.scope == "fragment"

    assert "fallback to container" not in caplog.text
    assert len(fake_st.fragment_calls) == 1
    assert fake_st.experimental_rerun.called is False

    messages = [record.getMessage() for record in caplog.records if "wait_for_fragment_context" in record.getMessage()]
    assert any("wait_for_fragment_context_start" in message for message in messages)
    assert any("wait_for_fragment_context_end" in message for message in messages)

    visibility_events = [event for event in telemetry_events if event.get("phase") == "portfolio.fragment_visibility"]
    assert visibility_events, "Debe registrarse la visibilidad del fragmento"
    payload = visibility_events[-1]
    extra = payload.get("extra", {})
    assert extra.get("lazy_loaded_component") == "table"
    assert extra.get("portfolio.fragment_visible") is True

