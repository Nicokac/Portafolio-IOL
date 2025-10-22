"""Tests for the notification banner rendered in :mod:`ui.header`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import pytest

import ui.header as header_module


class _DummyPlaceholder:
    def __init__(self, owner: "_DummyStreamlit") -> None:
        self._owner = owner
        self.info_calls: list[str] = []
        self.warning_calls: list[str] = []
        self.cleared = False

    def info(self, message: str) -> None:
        self.info_calls.append(message)
        self._owner.info(message)

    def warning(self, message: str) -> None:
        self.warning_calls.append(message)
        self._owner.warning(message)

    def empty(self) -> None:
        self.cleared = True


class _DummyStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.markdowns: list[str] = []
        self.placeholders: list[_DummyPlaceholder] = []

    def empty(self) -> _DummyPlaceholder:
        placeholder = _DummyPlaceholder(self)
        self.placeholders.append(placeholder)
        return placeholder

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def markdown(self, body: str, *, unsafe_allow_html: bool = False) -> None:
        self.markdowns.append(body)


@pytest.fixture(autouse=True)
def _reset_header_state(monkeypatch: pytest.MonkeyPatch) -> _DummyStreamlit:
    fake_st = _DummyStreamlit()
    fake_palette = SimpleNamespace(highlight_bg="#fff", highlight_text="#000")
    fake_cache = SimpleNamespace(get=lambda *_a, **_k: None)

    monkeypatch.setattr(header_module, "st", fake_st)
    monkeypatch.setattr(header_module, "get_active_palette", lambda: fake_palette)
    monkeypatch.setattr(header_module, "cache", fake_cache)

    # Ensure notification cache starts clean for each test.
    header_module.st.session_state.clear()
    return fake_st


def _mock_fetch(monkeypatch: pytest.MonkeyPatch, factory: Callable[[], dict | None]) -> None:
    monkeypatch.setattr(header_module, "_fetch_notification", lambda: factory())
    header_module.st.session_state.pop(header_module._NOTIFICATION_DATA_KEY, None)
    header_module.st.session_state.pop(header_module._NOTIFICATION_MARKER_KEY, None)


def test_render_header_shows_info_banner(
    monkeypatch: pytest.MonkeyPatch, _reset_header_state: _DummyStreamlit
) -> None:
    message = "Aviso general"

    def factory() -> dict | None:
        return {"mensaje": message, "activo": True}

    _mock_fetch(monkeypatch, factory)

    header_module.render_header()

    assert message in _reset_header_state.infos
    assert not _reset_header_state.warnings
    assert _reset_header_state.placeholders, "Se espera un placeholder para el banner"


def test_render_header_switches_to_warning(
    monkeypatch: pytest.MonkeyPatch, _reset_header_state: _DummyStreamlit
) -> None:
    message = "Mantenimiento programado"

    _mock_fetch(monkeypatch, lambda: {"mensaje": message, "activo": True})

    header_module.render_header()

    assert message in _reset_header_state.warnings
    assert not _reset_header_state.infos


def test_render_header_hides_banner_when_missing(
    monkeypatch: pytest.MonkeyPatch, _reset_header_state: _DummyStreamlit
) -> None:
    _mock_fetch(monkeypatch, lambda: None)

    header_module.render_header()

    assert not _reset_header_state.infos
    assert not _reset_header_state.warnings
    assert not _reset_header_state.placeholders


def test_render_header_reuses_placeholder(
    monkeypatch: pytest.MonkeyPatch, _reset_header_state: _DummyStreamlit
) -> None:
    message = "Aviso persistente"

    _mock_fetch(monkeypatch, lambda: {"mensaje": message, "activo": True})

    header_module.render_header()
    header_module.render_header()

    assert len(_reset_header_state.placeholders) == 1
    assert _reset_header_state.infos.count(message) >= 1


def test_fetch_notification_skips_without_tokens(_reset_header_state: _DummyStreamlit) -> None:
    result = header_module._fetch_notification()

    assert result is header_module._SKIP_NOTIFICATION_FETCH


def test_load_notification_preserves_cached_value_on_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    cached = {"mensaje": "Persistente"}
    header_module.st.session_state[header_module._NOTIFICATION_DATA_KEY] = cached
    header_module.st.session_state[header_module._NOTIFICATION_MARKER_KEY] = "prev"

    monkeypatch.setattr(header_module, "_current_notification_marker", lambda: "new")
    monkeypatch.setattr(header_module, "_can_attempt_notification_fetch", lambda: False)

    result = header_module._load_notification()

    assert result == cached
