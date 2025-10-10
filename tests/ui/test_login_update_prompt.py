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
        self.stopped = False

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

    def stop(self) -> None:
        self.stopped = True


def test_login_page_renders_update_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(login, "st", fake_st)
    monkeypatch.setattr(login, "render_header", lambda: None)
    monkeypatch.setattr(login, "render_footer", lambda: None)
    monkeypatch.setattr(login, "render_security_info", lambda: None)
    monkeypatch.setattr(
        login,
        "validate_tokens_key",
        lambda: SimpleNamespace(message=None, level="info", can_proceed=False),
    )
    monkeypatch.setattr(login, "check_for_update", lambda: "0.5.8")
    monkeypatch.setattr(login, "__version__", "0.5.7", raising=False)
    monkeypatch.setattr(login, "_run_update_script", lambda latest: True)

    login.render_login_page()

    assert fake_st.warnings, "Se espera que aparezca la advertencia de actualización"
    assert "Nueva versión disponible" in fake_st.warnings[-1]
    assert "0.5.8" in fake_st.warnings[-1]
    assert "Actualizar ahora" in fake_st.buttons
