from __future__ import annotations

import pytest

import ui.helpers.navigation as navigation

from ui.helpers.navigation import safe_page_link


def test_safe_page_link_runs_without_keyerror(monkeypatch, streamlit_stub) -> None:
    monkeypatch.setattr(navigation, "st", streamlit_stub)
    streamlit_stub.reset()
    streamlit_stub.runtime._page_registry = {}
    streamlit_stub.set_button_result("ℹ️ Acerca de", True)

    calls: list[str] = []

    def _render_fallback() -> None:
        calls.append("rendered")

    try:
        safe_page_link("ui.panels.about", "ℹ️ Acerca de", render_fallback=_render_fallback)
    except KeyError:
        pytest.fail("safe_page_link should handle KeyError gracefully")

    assert calls == ["rendered"]
    assert not streamlit_stub.get_records("page_link")
    buttons = streamlit_stub.get_records("button")
    assert any(entry["label"] == "ℹ️ Acerca de" for entry in buttons)


def test_safe_page_link_uses_registered_page(monkeypatch, streamlit_stub) -> None:
    monkeypatch.setattr(navigation, "st", streamlit_stub)
    streamlit_stub.reset()
    streamlit_stub.runtime._page_registry = {"ui.panels.about": object()}

    safe_page_link("ui.panels.about", "ℹ️ Acerca de")

    page_links = streamlit_stub.get_records("page_link")
    assert any(entry["page"] == "ui.panels.about" for entry in page_links)
    assert not streamlit_stub.get_records("button")


def test_safe_page_link_prefers_inline_when_requested(monkeypatch, streamlit_stub) -> None:
    monkeypatch.setattr(navigation, "st", streamlit_stub)
    streamlit_stub.reset()
    streamlit_stub.runtime._page_registry = {"ui.panels.about": object()}
    streamlit_stub.set_button_result("ℹ️ Acerca de", True)

    calls: list[str] = []

    def _render_inline() -> None:
        calls.append("inline")

    safe_page_link(
        "ui.panels.about",
        "ℹ️ Acerca de",
        render_fallback=_render_inline,
        prefer_inline=True,
    )

    assert calls == ["inline"]
    assert not streamlit_stub.get_records("page_link")
    buttons = streamlit_stub.get_records("button")
    assert any(entry["label"] == "ℹ️ Acerca de" for entry in buttons)
