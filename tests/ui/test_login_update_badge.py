from __future__ import annotations

from types import SimpleNamespace

import pytest

import ui.login as login


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self.buttons: list[str] = []
        self.statuses: list[tuple[str, str]] = []
        self.captions: list[str] = []
        self.warnings: list[str] = []
        self.successes: list[str] = []
        self.link_buttons: list[tuple[str, str]] = []
        self.stopped = False
        self._button_responses: dict[str, list[bool]] = {}

    def markdown(self, *_args, **_kwargs) -> None:
        return None

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

    def success(self, message: str) -> None:
        self.successes.append(message)

    def link_button(self, label: str, url: str) -> None:
        self.link_buttons.append((label, url))

    def status(self, label: str, *, state: str) -> None:
        self.statuses.append((label, state))

    class _Expander:
        def __enter__(self) -> "_FakeStreamlit._Expander":
            return self

        def __exit__(self, *_args) -> None:
            return None

    def expander(self, _label: str) -> "_FakeStreamlit._Expander":
        return self._Expander()

    def stop(self) -> None:
        self.stopped = True


def _setup_common(monkeypatch: pytest.MonkeyPatch, fake_st: _FakeStreamlit) -> None:
    monkeypatch.setattr(login, "st", fake_st)
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
    monkeypatch.setattr(login, "__version__", "0.5.9", raising=False)


def test_update_button_displays_running_badge(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = _FakeStreamlit()
    fake_st.queue_button_response("Actualizar ahora", True)

    _setup_common(monkeypatch, fake_st)
    monkeypatch.setattr(login, "check_for_update", lambda: "0.6.0")
    monkeypatch.setattr(login, "get_update_history", lambda: [])

    calls: list[str] = []
    monkeypatch.setattr(login, "_run_update_script", lambda version: calls.append(version))

    login.render_login_page()

    assert calls == ["0.6.0"]
    assert ("Actualizando aplicación...", "running") in fake_st.statuses
    assert ("Actualización completada", "complete") in fake_st.statuses
    assert fake_st.stopped is True


def test_history_panel_renders_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = _FakeStreamlit()

    _setup_common(monkeypatch, fake_st)
    monkeypatch.setattr(login, "check_for_update", lambda: None)
    monkeypatch.setattr(
        login,
        "get_update_history",
        lambda: [
            {
                "timestamp": "2024-05-01 10:00:00",
                "event": "check",
                "version": "0.5.9",
                "status": "ok",
            },
            {
                "timestamp": "2024-05-02 12:30:00",
                "event": "update",
                "version": "0.5.9",
                "status": "done",
            },
        ],
    )

    login.render_login_page()

    assert any("check v0.5.9" in caption for caption in fake_st.captions)
    assert any("update v0.5.9" in caption for caption in fake_st.captions)
