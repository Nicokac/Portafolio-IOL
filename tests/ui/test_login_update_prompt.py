from __future__ import annotations

from types import SimpleNamespace

import pytest

import ui.login as login


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self.warnings: list[str] = []
        self.buttons: list[str] = []
        self.captions: list[str] = []
        self.infos: list[str] = []
        self.successes: list[str] = []
        self.link_buttons: list[tuple[str, str]] = []
        self.statuses: list[tuple[str, str]] = []
        self.stopped = False
        self.sidebar = self._Sidebar()

    def markdown(self, *_args, **_kwargs) -> None:
        return None

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def button(self, label: str) -> bool:
        self.buttons.append(label)
        return False

    def caption(self, message: str) -> None:
        self.captions.append(message)

    def info(self, message: str) -> None:
        self.infos.append(message)

    def success(self, message: str) -> None:
        self.successes.append(message)

    def link_button(self, label: str, url: str) -> None:
        self.link_buttons.append((label, url))

    def status(self, label: str, *, state: str) -> None:
        self.statuses.append((label, state))

    def checkbox(
        self,
        label: str,
        *,
        value: bool = False,
        key: str | None = None,
        disabled: bool = False,
    ) -> bool:
        if key is not None and key not in self.session_state:
            self.session_state[key] = value
        return value and not disabled

    class _Sidebar:
        def __enter__(self) -> "_FakeStreamlit._Sidebar":
            return self

        def __exit__(self, *_args) -> None:
            return None

    class _Expander:
        def __enter__(self) -> "_FakeStreamlit._Expander":
            return self

        def __exit__(self, *_args) -> None:
            return None

    def expander(self, _label: str) -> "_FakeStreamlit._Expander":
        return self._Expander()

    def stop(self) -> None:
        self.stopped = True


def _patch_login_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(login, "render_header", lambda: None)
    monkeypatch.setattr(login, "render_footer", lambda: None)
    monkeypatch.setattr(login, "render_security_info", lambda: None)
    monkeypatch.setattr(
        login,
        "validate_tokens_key",
        lambda: SimpleNamespace(message=None, level="info", can_proceed=False),
    )
    monkeypatch.setattr(login, "get_last_check_time", lambda: None)
    monkeypatch.setattr(login, "format_last_check", lambda _ts: "Nunca")
    monkeypatch.setattr(login, "get_update_history", lambda: [])


def test_login_page_renders_update_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(login, "st", fake_st)
    _patch_login_dependencies(monkeypatch)
    monkeypatch.setattr(
        login,
        "safe_page_link",
        lambda page, label, render_fallback: None,
    )
    monkeypatch.setattr(login, "check_for_update", lambda: "0.5.8")
    monkeypatch.setattr(login, "__version__", "0.5.7", raising=False)
    monkeypatch.setattr(login, "_run_update_script", lambda latest: True)

    login.render_login_page()

    assert fake_st.warnings, "Se espera que aparezca la advertencia de actualización"
    assert "Nueva versión disponible" in fake_st.warnings[-1]
    assert any(
        label == "📄 Ver cambios en GitHub" for label, _ in fake_st.link_buttons
    ), "Se espera un enlace al changelog"
    assert "0.5.8" in fake_st.warnings[-1]
    assert "Actualizar ahora" in fake_st.buttons


def test_login_about_link_registered(monkeypatch: pytest.MonkeyPatch, streamlit_stub) -> None:
    streamlit_stub.reset()
    streamlit_stub.runtime._page_registry = {"ui.panels.about": object()}
    monkeypatch.setattr(login, "st", streamlit_stub, raising=False)
    _patch_login_dependencies(monkeypatch)
    monkeypatch.setattr(login, "check_for_update", lambda: None)

    login.render_login_page()

    page_links = streamlit_stub.get_records("page_link")
    assert any(entry["page"] == "ui.panels.about" for entry in page_links)


def test_login_about_link_inline_fallback(monkeypatch: pytest.MonkeyPatch, streamlit_stub) -> None:
    streamlit_stub.reset()
    streamlit_stub.runtime._page_registry = {}
    streamlit_stub.set_button_result("ℹ️ Acerca de", True)
    monkeypatch.setattr(login, "st", streamlit_stub, raising=False)
    _patch_login_dependencies(monkeypatch)
    monkeypatch.setattr(login, "check_for_update", lambda: None)

    fallback_calls: list[str] = []
    monkeypatch.setattr(login, "render_about_panel", lambda: fallback_calls.append("rendered"))

    login.render_login_page()

    assert fallback_calls == ["rendered"]
    assert not streamlit_stub.get_records("page_link")
    buttons = streamlit_stub.get_records("button")
    assert any(entry["label"] == "ℹ️ Acerca de" for entry in buttons)
