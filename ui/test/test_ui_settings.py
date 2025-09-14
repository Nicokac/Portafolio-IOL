import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import streamlit as st
import ui.ui_settings as ui_settings
from ui.ui_settings import UISettings, apply_settings, get_settings


def test_get_settings_returns_defaults(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    settings = get_settings()
    assert settings.layout == "wide"
    assert settings.theme == "dark"


def test_get_settings_reads_session_state(monkeypatch):
    monkeypatch.setattr(
        st, "session_state", {"ui_layout": "centered", "ui_theme": "light"}
    )
    settings = get_settings()
    assert settings.layout == "centered"
    assert settings.theme == "light"


def test_apply_settings_invokes_set_page_config(monkeypatch):
    monkeypatch.setattr(st, "set_page_config", MagicMock())
    monkeypatch.setattr(st, "markdown", MagicMock())
    apply_settings(UISettings(layout="centered", theme="dark"))
    st.set_page_config.assert_called_once_with(
        page_title="IOL — Portafolio en vivo (solo lectura)",
        layout="centered",
    )


def test_render_ui_controls_reruns_on_changes(monkeypatch):
    state = {"ui_layout": "wide", "ui_theme": "dark"}

    def expander(label):
        return contextlib.nullcontext()

    def radio(label, options, index=0):
        return {"Layout": "centered", "Tema": "light"}[label]

    rerun_called = {"called": False}

    def rerun():
        rerun_called["called"] = True

    mock_st = SimpleNamespace(
        session_state=state,
        sidebar=SimpleNamespace(expander=expander),
        radio=radio,
        rerun=rerun,
        set_page_config=MagicMock(),
        markdown=MagicMock(),
    )
    monkeypatch.setattr(ui_settings, "st", mock_st)

    settings = ui_settings.render_ui_controls()
    assert rerun_called["called"] is True
    assert state["ui_layout"] == "centered"
    assert state["ui_theme"] == "light"
    assert settings.layout == "centered"
    assert settings.theme == "light"
