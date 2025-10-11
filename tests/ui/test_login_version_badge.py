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


def test_login_page_shows_version_badge(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(
        login,
        "safe_page_link",
        lambda page, label, render_fallback: None,
    )
    monkeypatch.setattr(login, "check_for_update", lambda: None)
    monkeypatch.setattr(login, "get_last_check_time", lambda: None)
    monkeypatch.setattr(login, "format_last_check", lambda _ts: "Nunca")

    login.render_login_page()

    assert (
        f"Versión actualizada · v{login.__version__}",
        "complete",
    ) in fake_st.statuses
    assert any(
        label == "📄 Ver cambios en GitHub" for label, _ in fake_st.link_buttons
    ), "El enlace al changelog debe estar siempre presente"
    assert any(
        msg.startswith("Última verificación:") for msg in fake_st.captions
    ), "Debe mostrar la última verificación"
    assert "Forzar actualización" in fake_st.buttons
