# ui/ui_settings.py
from __future__ import annotations
from dataclasses import dataclass
import streamlit as st

@dataclass
class UISettings:
    """Simple container for UI configuration."""
    layout: str = "wide"  # "wide" or "centered"
    theme: str = "light"  # "light" or "dark"


def get_settings() -> UISettings:
    """Return current UI settings from session state or defaults."""
    return UISettings(
        layout=st.session_state.get("ui_layout", "wide"),
        theme=st.session_state.get("ui_theme", "light"),
    )


def apply_settings(settings: UISettings) -> None:
    """Apply settings to the Streamlit page."""
    st.set_page_config(
        page_title="IOL — Portafolio en vivo (solo lectura)",
        layout=settings.layout,
    )
    # Streamlit aún no expone toggle de tema vía API, así que usamos CSS simple
    if settings.theme == "dark":
        st.markdown(
            """
            <style>
            :root { color-scheme: dark; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            :root { color-scheme: light; }
            </style>
            """,
            unsafe_allow_html=True,
        )


def init_ui() -> UISettings:
    """Convenience helper to obtain current settings and apply them."""
    s = get_settings()
    apply_settings(s)
    return s


def render_ui_controls() -> UISettings:
    """Render controls in sidebar allowing the user to tweak layout/theme."""
    current = get_settings()
    with st.sidebar.expander("Apariencia"):
        layout = st.radio(
            "Layout",
            ("wide", "centered"),
            index=0 if current.layout == "wide" else 1,
        )
        theme = st.radio(
            "Tema",
            ("light", "dark"),
            index=0 if current.theme == "light" else 1,
        )
    if layout != current.layout or theme != current.theme:
        st.session_state["ui_layout"] = layout
        st.session_state["ui_theme"] = theme
        st.rerun()
    return get_settings()