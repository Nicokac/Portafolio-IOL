"""UI helpers to highlight risk, technical signals and upcoming earnings."""
from __future__ import annotations

import html
from typing import Literal

import streamlit as st

BadgeVariant = Literal["risk", "technical", "earnings"]

_BADGE_STYLE_KEY = "_notification_badge_css"

_BADGE_CONFIG: dict[BadgeVariant, dict[str, str]] = {
    "risk": {
        "label": "Alerta de riesgo",
        "icon": "⚠️",
        "suffix": " ⚠️",
    },
    "technical": {
        "label": "Señales técnicas activas",
        "icon": "📈",
        "suffix": " 📈",
    },
    "earnings": {
        "label": "Earnings próximos",
        "icon": "🗓️",
        "suffix": " 🗓️",
    },
}


def _ensure_badge_styles() -> None:
    if st.session_state.get(_BADGE_STYLE_KEY):
        return
    st.markdown(
        """
        <style>
            .notification-badge-row {
                margin-top: 0.2rem;
                margin-bottom: 0.75rem;
                display: flex;
                gap: 0.5rem;
                flex-wrap: wrap;
            }
            .notification-badge {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                font-weight: 600;
                font-size: 0.82rem;
                padding: 0.25rem 0.75rem;
                border: 1px solid transparent;
                box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.2);
            }
            .notification-badge__icon {
                margin-right: 0.45rem;
            }
            .notification-badge__label {
                white-space: nowrap;
            }
            .notification-badge--risk {
                background-color: #fee2e2;
                color: #b91c1c;
                border-color: #fecaca;
            }
            .notification-badge--technical {
                background-color: #e0f2fe;
                color: #1d4ed8;
                border-color: #bae6fd;
            }
            .notification-badge--earnings {
                background-color: #fef3c7;
                color: #b45309;
                border-color: #fde68a;
            }
            .notification-badge__wrapper {
                display: inline-flex;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[_BADGE_STYLE_KEY] = True


def _build_badge_html(variant: BadgeVariant, *, text: str | None = None, help_text: str | None = None) -> str:
    _ensure_badge_styles()
    config = _BADGE_CONFIG[variant]
    label = html.escape(text or config["label"])  # noqa: S703 - html escape is explicit
    icon = html.escape(config["icon"])  # noqa: S703 - html escape is explicit
    badge = (
        "<span class='notification-badge notification-badge--{variant}'>"
        "<span class='notification-badge__icon'>{icon}</span>"
        "<span class='notification-badge__label'>{label}</span>"
        "</span>"
    ).format(variant=variant, icon=icon, label=label)
    if help_text:
        badge = (
            "<span class='notification-badge__wrapper' title='{title}'>"
            "{badge}</span>"
        ).format(title=html.escape(help_text), badge=badge)
    return badge


def render_notification_badge(
    variant: BadgeVariant,
    *,
    text: str | None = None,
    help_text: str | None = None,
) -> None:
    """Render a notification badge for the requested variant."""

    badge_html = _build_badge_html(variant, text=text, help_text=help_text)
    st.markdown(
        "<div class='notification-badge-row'>{badge}</div>".format(badge=badge_html),
        unsafe_allow_html=True,
    )


def render_risk_badge(*, help_text: str | None = None) -> None:
    render_notification_badge("risk", help_text=help_text)


def render_technical_badge(*, help_text: str | None = None) -> None:
    render_notification_badge("technical", help_text=help_text)


def render_earnings_badge(*, help_text: str | None = None) -> None:
    render_notification_badge("earnings", help_text=help_text)


def tab_badge_suffix(variant: BadgeVariant) -> str:
    """Return the suffix appended to the tab label when a badge is active."""

    return _BADGE_CONFIG[variant]["suffix"]


def tab_badge_label(variant: BadgeVariant) -> str:
    """Return the human readable label for the badge variant."""

    return _BADGE_CONFIG[variant]["label"]


__all__ = [
    "render_notification_badge",
    "render_risk_badge",
    "render_technical_badge",
    "render_earnings_badge",
    "tab_badge_suffix",
    "tab_badge_label",
]
