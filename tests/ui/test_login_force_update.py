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
        self.page_links: list[tuple[str, str]] = []
        self.checkboxes: list[tuple[str, bool, str, bool]] = []
        self.markdowns: list[tuple[tuple, dict]] = []
        self._button_responses: dict[str, list[bool]] = {}
        self.stopped = False
        self.sidebar = self._Sidebar(self)

    class _Sidebar:
        def __init__(self, parent: "_FakeStreamlit") -> None:
            self.parent = parent

        def __enter__(self) -> "_FakeStreamlit._Sidebar":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def markdown(self, *args, **kwargs) -> None:
        self.markdowns.append((args, kwargs))

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def button(self, label: str) -> bool:
        self.buttons.append(label)
        responses = self._button_responses.get(label, [])
        return responses.pop(0) if responses else False

    def queue_button_response(self, label: str, value: bool) -> None:
        self._button_responses.setdefault(label, []).append(value)

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

    def checkbox(self, label: str, *, value: bool, key: str, disabled: bool = False) -> bool:
        self.checkboxes.append((label, value, key, disabled))
        if disabled:
            self.session_state[key] = value
            return value
        if key in self.session_state:
            return bool(self.session_state[key])
        self.session_state[key] = value
        return value

    def page_link(self, page: str, *, label: str) -> None:
        self.page_links.append((page, label))

    class _Expander:
        def __enter__(self) -> "_FakeStreamlit._Expander":
            return self

        def __exit__(self, *_args) -> None:
            return None

    def expander(self, _label: str) -> "_FakeStreamlit._Expander":
        return self._Expander()

    def stop(self) -> None:
        self.stopped = True


def test_force_update_triggers_run_update(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = _FakeStreamlit()
    fake_st.queue_button_response("Forzar actualización", True)
    fake_st.queue_button_response("Confirmar actualización", True)

    monkeypatch.setattr(login, "st", fake_st)
    monkeypatch.setattr(login, "render_header", lambda: None)
    monkeypatch.setattr(login, "render_footer", lambda: None)
    monkeypatch.setattr(login, "render_security_info", lambda: None)
    monkeypatch.setattr(
        login,
        "validate_tokens_key",
        lambda: SimpleNamespace(message=None, level="info", can_proceed=False),
    )
    monkeypatch.setattr(
        login,
        "safe_page_link",
        lambda page, label, render_fallback: fake_st.page_link(page, label=label),
    )
    monkeypatch.setattr(login, "check_for_update", lambda: None)
    monkeypatch.setattr(login, "get_last_check_time", lambda: None)
    monkeypatch.setattr(login, "format_last_check", lambda _ts: "Nunca")
    monkeypatch.setattr(login, "get_update_history", lambda: [])
    monkeypatch.setattr(login, "__version__", "0.5.8", raising=False)

    calls: list[str] = []

    def fake_run(version: str) -> bool:
        calls.append(version)
        return True

    monkeypatch.setattr(login, "_run_update_script", fake_run)
    restart_calls: list[None] = []
    monkeypatch.setattr(login, "safe_restart_app", lambda: restart_calls.append(None))

    login.render_login_page()

    assert "Forzar actualización" in fake_st.buttons
    assert "Confirmar actualización" in fake_st.buttons
    assert calls == ["0.5.8"]
    assert ("Actualizando aplicación...", "running") in fake_st.statuses
    assert ("Actualización completada", "complete") in fake_st.statuses
    assert fake_st.successes[-1] == "✅ Actualización completada. Reiniciando..."
    assert "🔁 Reiniciando aplicación..." in fake_st.infos
    assert restart_calls == [None]
    assert ("ui.panels.about", "ℹ️ Acerca de") in fake_st.page_links
    assert fake_st.stopped is True
