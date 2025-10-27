# ui/ui_settings.py
from dataclasses import dataclass

import streamlit as st

from shared.debug.rerun_trace import mark_event, safe_rerun

from .palette import get_palette
from .utils.bootstrap import ensure_bootstrap_assets


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


_PAGE_CONFIGURED_KEY = "_ui_page_configured"


def _ensure_page_config(layout: str) -> None:
    if not hasattr(st, "set_page_config"):
        return
    try:
        if st.session_state.get(_PAGE_CONFIGURED_KEY):
            return
    except Exception:  # pragma: no cover - session state may be read-only
        pass
    try:
        st.set_page_config(
            page_title="IOL — Portafolio en vivo (solo lectura)",
            layout=layout or "wide",
        )
    except Exception:  # pragma: no cover - defensive guard for Streamlit stubs
        return
    try:
        st.session_state[_PAGE_CONFIGURED_KEY] = True
    except Exception:  # pragma: no cover - session state may be read-only
        pass


def apply_settings(settings: UISettings) -> None:
    """Apply settings to the Streamlit page."""
    ensure_bootstrap_assets()
    _ensure_page_config(settings.layout)
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
            --color-highlight-bg: {pal.highlight_bg};
            --color-highlight-text: {pal.highlight_text};
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


_LAYOUT_WIDGET_KEY = "_ui_controls_layout"
_THEME_WIDGET_KEY = "_ui_controls_theme"


def _sync_setting(source_key: str, target_key: str) -> None:
    try:
        value = st.session_state.get(source_key)
    except Exception:  # pragma: no cover - session state may be read-only
        value = None
    if value is not None:
        try:
            st.session_state[target_key] = value
        except Exception:  # pragma: no cover - defensive guard
            pass
    try:
        mark_event("rerun", "ui_settings_sync")
        safe_rerun("ui_settings_sync", rerun_args=(False,))
    except (RuntimeError, TypeError):
        mark_event("rerun", "ui_settings_sync")
        safe_rerun("ui_settings_sync")
    except Exception:  # pragma: no cover - defensive guard
        pass


def render_ui_controls(container=None) -> UISettings:
    """Render controls allowing the user to tweak layout/theme."""

    host = container if container is not None else st

    current = get_settings()

    if hasattr(host, "markdown"):
        host.markdown("### Apariencia")

    try:
        st.session_state.setdefault(_LAYOUT_WIDGET_KEY, current.layout)
        st.session_state.setdefault(_THEME_WIDGET_KEY, current.theme)
    except Exception:  # pragma: no cover - session state may be read-only
        pass

    try:
        host.radio(
            "Layout",
            ("wide", "centered"),
            index=0 if current.layout == "wide" else 1,
            key=_LAYOUT_WIDGET_KEY,
            on_change=_sync_setting,
            args=(_LAYOUT_WIDGET_KEY, "ui_layout"),
        )
    except TypeError:
        selection = host.radio(
            "Layout",
            ("wide", "centered"),
            index=0 if current.layout == "wide" else 1,
        )
        try:
            st.session_state[_LAYOUT_WIDGET_KEY] = selection
        except Exception:
            pass
        _sync_setting(_LAYOUT_WIDGET_KEY, "ui_layout")

    try:
        host.radio(
            "Tema",
            ("light", "dark"),
            index=0 if current.theme == "light" else 1,
            key=_THEME_WIDGET_KEY,
            on_change=_sync_setting,
            args=(_THEME_WIDGET_KEY, "ui_theme"),
        )
    except TypeError:
        selection = host.radio(
            "Tema",
            ("light", "dark"),
            index=0 if current.theme == "light" else 1,
        )
        try:
            st.session_state[_THEME_WIDGET_KEY] = selection
        except Exception:
            pass
        _sync_setting(_THEME_WIDGET_KEY, "ui_theme")

    return get_settings()
