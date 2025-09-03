# ui/ui_settings.py
from __future__ import annotations
from dataclasses import dataclass
import streamlit as st
import time
from infrastructure.iol.auth import IOLAuth
from .palette import get_palette

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
    pal = get_palette(settings.theme)
    st.markdown(
        f"""
        <style>
        :root {{
            color-scheme: {settings.theme};
            --color-bg: {pal.bg};
            --color-text: {pal.text};
            --color-positive: {pal.positive};
            --color-negative: {pal.negative};
            --color-accent: {pal.accent};
        }}
        html, body, [data-testid="stAppViewContainer"] {{
            background-color: var(--color-bg);
            color: var(--color-text);
        }}
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

def render_action_menu(user: str, password: str) -> None:
    """Render refresh and relogin actions in a compact popover."""
    pop = st.popover("⚙️ Acciones")
    with pop:
        st.caption("Operaciones rápidas")
        c1, c2 = st.columns(2)
        if c1.button("⟳ Refrescar", use_container_width=True):
            st.session_state["refresh_pending"] = True
            st.rerun()
        if c2.button("🔄 Relogin", use_container_width=True):
            st.session_state["relogin_pending"] = True
            st.rerun()

    if st.session_state.pop("refresh_pending", False):
        with st.spinner("Actualizando datos..."):
            st.session_state["last_refresh"] = time.time()
        st.session_state["show_refresh_toast"] = True
        st.rerun()

    if st.session_state.pop("relogin_pending", False):
        with st.spinner("Eliminando tokens..."):
            try:
                IOLAuth(user, password).clear_tokens()
                st.session_state["client_salt"] = int(time.time())
                st.session_state["relogin_done"] = True
            except Exception as e:
                st.session_state["relogin_error"] = str(e)
        st.rerun()

    if st.session_state.pop("show_refresh_toast", False):
        st.toast("Datos actualizados", icon="✅")

    if st.session_state.pop("relogin_done", False):
        st.success("Tokens eliminados. Recargando…")

    err = st.session_state.pop("relogin_error", "")
    if err:
        st.error(f"No se pudo limpiar tokens: {err}")
        st.stop()