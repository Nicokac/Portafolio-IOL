"""Validation tests for the login lazy-startup flow."""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest

import ui.login as login


class _FormStub:
    def __enter__(self) -> "_FormStub":  # pragma: no cover - context protocol
        return self

    def __exit__(self, *_args) -> None:  # pragma: no cover - context protocol
        return None


class _ColumnStub:
    def __enter__(self) -> "_ColumnStub":
        return self

    def __exit__(self, *_args) -> None:
        return None


class _ExpanderStub:
    def __enter__(self) -> "_ExpanderStub":
        return self

    def __exit__(self, *_args) -> None:
        return None


class _SidebarStub:
    def __enter__(self) -> "_SidebarStub":
        return self

    def __exit__(self, *_args) -> None:
        return None


class _StreamlitStub:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self.sidebar = _SidebarStub()

    def markdown(self, *_args, **_kwargs) -> None:
        return None

    def checkbox(self, _label: str, value: bool = False, **_kwargs) -> bool:
        return value

    def button(self, _label: str) -> bool:
        return False

    def status(self, *_args, **_kwargs) -> None:
        return None

    def caption(self, *_args, **_kwargs) -> None:
        return None

    def link_button(self, *_args, **_kwargs) -> None:
        return None

    def warning(self, *_args, **_kwargs) -> None:
        return None

    def error(self, *_args, **_kwargs) -> None:
        return None

    def success(self, *_args, **_kwargs) -> None:
        return None

    def info(self, *_args, **_kwargs) -> None:
        return None

    def form(self, _name: str) -> _FormStub:
        return _FormStub()

    def form_submit_button(self, _label: str) -> bool:
        return False

    def text_input(self, _label: str, **_kwargs) -> str:
        return ""

    def columns(self, spec) -> tuple[_ColumnStub, ...]:
        if isinstance(spec, int):
            count = max(1, spec)
        else:
            count = len(tuple(spec))
        return tuple(_ColumnStub() for _ in range(count))

    def expander(self, _label: str) -> _ExpanderStub:
        return _ExpanderStub()

    def stop(self) -> None:  # pragma: no cover - compatibility
        raise RuntimeError("stop called")


@pytest.fixture
def streamlit_stub(monkeypatch: pytest.MonkeyPatch) -> _StreamlitStub:
    stub = _StreamlitStub()
    monkeypatch.setattr(login, "st", stub)
    return stub


def _setup_login_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(login, "render_security_info", lambda: None)
    monkeypatch.setattr(
        login,
        "_lazy_attr",
        lambda module, attr: (lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(login, "check_for_update", lambda: None)
    monkeypatch.setattr(login, "get_update_history", lambda: [])
    monkeypatch.setattr(login, "get_last_check_time", lambda: None)
    monkeypatch.setattr(login, "format_last_check", lambda _ts: "Nunca")
    monkeypatch.setattr(
        login,
        "validate_tokens_key",
        lambda: SimpleNamespace(message=None, level="info", can_proceed=True),
    )
    monkeypatch.setattr(login, "_is_fastapi_available", lambda: False)
    monkeypatch.setattr(login, "_is_engine_api_active", lambda: False)


def test_login_render_is_fast_and_lazy(monkeypatch: pytest.MonkeyPatch, streamlit_stub: _StreamlitStub) -> None:
    _setup_login_stubs(monkeypatch)

    streamlit_stub.session_state["_security_validated"] = True

    sys.modules.pop("services.performance_timer", None)
    sys.modules.pop("services.cache.market_data_cache", None)

    start = time.perf_counter()
    login.render_login_page()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert elapsed_ms < 2500
    assert "services.performance_timer" not in sys.modules
    assert "services.cache.market_data_cache" not in sys.modules
    assert streamlit_stub.session_state.get("_security_validated") is True
