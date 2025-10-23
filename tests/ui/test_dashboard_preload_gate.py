from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType

import pytest

import services.preload_worker as preload
import ui.helpers.preload as preload_helper


def _install_ui_stubs() -> None:
    dummy_modules: dict[str, dict[str, object]] = {
        "header": {"render_header": lambda *args, **kwargs: None},
        "tables": {
            "render_table": lambda *args, **kwargs: None,
        },
        "summary_metrics": {
            "render_summary_metrics": lambda *args, **kwargs: None,
        },
        "fx_panels": {
            "render_spreads": lambda *args, **kwargs: None,
            "render_fx_history": lambda *args, **kwargs: None,
        },
        "sidebar_controls": {"render_sidebar": lambda *args, **kwargs: None},
        "fundamentals": {
            "render_fundamental_data": lambda *args, **kwargs: None,
            "render_fundamental_ranking": lambda *args, **kwargs: None,
            "render_sector_comparison": lambda *args, **kwargs: None,
        },
        "ui_settings": {
            "init_ui": lambda *args, **kwargs: None,
            "render_ui_controls": lambda *args, **kwargs: None,
            "UISettings": object,
        },
        "actions": {"render_action_menu": lambda *args, **kwargs: None},
        "palette": {
            "get_palette": lambda *args, **kwargs: None,
            "get_active_palette": lambda *args, **kwargs: None,
        },
        "footer": {"render_footer": lambda *args, **kwargs: None},
    }

    for name, attrs in dummy_modules.items():
        module_name = f"ui.{name}"
        module = ModuleType(module_name)
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
        sys.modules[module_name] = module


_install_ui_stubs()


@pytest.fixture(autouse=True)
def restore_session_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(preload_helper.st, "session_state", {}, raising=False)
    yield
    preload_helper.st.session_state.clear()


@contextmanager
def _dummy_spinner(message: str):
    yield


class _Placeholder:
    def __init__(self):
        self.cleared = False
        self.entered = False

    def container(self):
        self.entered = True
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def empty(self):
        self.cleared = True


class _Container:
    def __init__(self):
        self.calls = 0

    def empty(self):
        self.calls += 1
        return _Placeholder()


def test_ensure_scientific_preload_ready_waits_for_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = preload_helper.st.session_state
    state.clear()
    ready_flag = {"done": False}

    monkeypatch.setattr(preload, "is_preload_complete", lambda: ready_flag["done"])

    def fake_wait(timeout: float | None = None) -> bool:
        ready_flag["done"] = True
        return True

    monkeypatch.setattr(preload, "wait_for_preload_completion", fake_wait)
    monkeypatch.setattr(preload_helper.st, "spinner", lambda message: _dummy_spinner(message))

    container = _Container()
    result = preload_helper.ensure_scientific_preload_ready(container)

    assert result is True
    assert ready_flag["done"] is True
    assert state["scientific_preload_ready"] is True
    assert container.calls == 1


def test_ensure_scientific_preload_ready_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preload, "is_preload_complete", lambda: True)
    container = _Container()
    result = preload_helper.ensure_scientific_preload_ready(container)
    assert result is True
    assert container.calls == 0
    assert preload_helper.st.session_state["scientific_preload_ready"] is True


def test_ensure_scientific_preload_ready_handles_stub_module(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    stub = ModuleType("services.preload_worker")
    monkeypatch.setitem(sys.modules, "services.preload_worker", stub)
    caplog.set_level("WARNING")

    container = _Container()
    result = preload_helper.ensure_scientific_preload_ready(container)

    assert result is False
    assert preload_helper.st.session_state["scientific_preload_ready"] is False
    assert "[preload]" in caplog.text
    assert container.calls == 0


def test_ensure_scientific_preload_ready_times_out(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    state = preload_helper.st.session_state
    state.clear()

    monkeypatch.setattr(preload, "is_preload_complete", lambda: False)

    def fake_wait(timeout: float | None = None) -> bool:
        return False

    monkeypatch.setattr(preload, "wait_for_preload_completion", fake_wait)
    monkeypatch.setattr(preload_helper.st, "spinner", lambda message: _dummy_spinner(message))

    container = _Container()
    with caplog.at_level("WARNING"):
        result = preload_helper.ensure_scientific_preload_ready(container, timeout_seconds=0.1)

    assert result is False
    assert preload_helper.st.session_state["scientific_preload_ready"] is False
    assert container.calls == 1
    assert "La precarga científica no finalizó" in caplog.text
