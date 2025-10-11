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
        self._button_responses: dict[str, list[bool]] = {}
        self.stopped = False

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

    def info(self, message: str) -> None:
        self.infos.append(message)

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
    monkeypatch.setattr(login, "check_for_update", lambda: None)
    monkeypatch.setattr(login, "get_last_check_time", lambda: None)
    monkeypatch.setattr(login, "format_last_check", lambda _ts: "Nunca")
    monkeypatch.setattr(login, "__version__", "0.5.8", raising=False)

    calls: list[str] = []
    monkeypatch.setattr(login, "_run_update_script", lambda version: calls.append(version))

    login.render_login_page()

    assert "Forzar actualización" in fake_st.buttons
    assert "Confirmar actualización" in fake_st.buttons
    assert calls == ["0.5.8"]
    assert fake_st.stopped is True
