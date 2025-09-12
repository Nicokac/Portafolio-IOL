import streamlit as st
from ui.ui_settings import get_settings


def test_default_theme_is_dark():
    st.session_state.clear()
    settings = get_settings()
    assert settings.theme == "dark"


def test_manual_light_theme_persists():
    st.session_state.clear()
    st.session_state["ui_theme"] = "light"
    settings = get_settings()
    assert settings.theme == "light"
    # ensure session state retains the user choice
    assert st.session_state["ui_theme"] == "light"
