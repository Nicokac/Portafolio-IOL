# ui/ui_settings.py
from dataclasses import dataclass
import streamlit as st
from .palette import get_palette

@dataclass
class UISettings:
    """Simple container for UI configuration."""
    layout: str = "wide"  # "wide" or "centered"
    theme: str = "dark"  # "light" or "dark"


def get_settings() -> UISettings:
    """Return current UI settings from session state or defaults."""
    return UISettings(
        layout=st.session_state.get("ui_layout", "wide"),
        theme=st.session_state.get("ui_theme", "dark"),
    )


def apply_settings(settings: UISettings) -> None:
    """Apply settings to the Streamlit page."""
    if hasattr(st, "set_page_config"):
        st.set_page_config(
            page_title="IOL — Portafolio en vivo (solo lectura)",
            layout=settings.layout,
        )
    pal = get_palette(settings.theme)
    style_block = f"""
        <style>
        :root {{
            color-scheme: {settings.theme};
            --color-bg: {pal.bg};
            --color-text: {pal.text};
            --color-positive: {pal.positive};
            --color-negative: {pal.negative};
            --color-accent: {pal.accent};
        }}
        html, body, [data-testid=\"stAppViewContainer\"] {{
            background-color: var(--color-bg);
            color: var(--color-text);
        }}
        </style>
    """
    try:
        st.markdown(style_block, unsafe_allow_html=True)
    except TypeError:
        st.markdown(style_block)

def init_ui() -> UISettings:
    """Convenience helper to obtain current settings and apply them."""
    s = get_settings()
    apply_settings(s)
    return s


def render_ui_controls(container=None) -> UISettings:
    """Render controls allowing the user to tweak layout/theme."""

    host = container if container is not None else st

    current = get_settings()

    if hasattr(host, "markdown"):
        host.markdown("### Apariencia")

    layout = host.radio(
        "Layout",
        ("wide", "centered"),
        index=0 if current.layout == "wide" else 1,
    )
    theme = host.radio(
        "Tema",
        ("light", "dark"),
        index=0 if current.theme == "light" else 1,
    )

    new_layout = layout if isinstance(layout, str) else current.layout
    new_theme = theme if isinstance(theme, str) else current.theme

    if new_layout != current.layout or new_theme != current.theme:
        st.session_state["ui_layout"] = new_layout
        st.session_state["ui_theme"] = new_theme
        st.rerun()

    return get_settings()
