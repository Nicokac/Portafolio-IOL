"""UI helpers to highlight risk, technical signals and upcoming earnings."""

from __future__ import annotations

import html
from typing import Literal

import streamlit as st

BadgeVariant = Literal["risk", "technical", "earnings"]

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


def _build_badge_html(variant: BadgeVariant, *, text: str | None = None, help_text: str | None = None) -> str:
    config = _BADGE_CONFIG[variant]
    label = html.escape(text or config["label"])
    icon = html.escape(config["icon"])
    badge = (
        "<span class='notification-badge notification-badge--{variant}'>"
        "<span class='notification-badge__icon'>{icon}</span>"
        "<span class='notification-badge__label'>{label}</span>"
        "</span>"
    ).format(variant=variant, icon=icon, label=label)
    if help_text:
        badge = ("<span class='notification-badge__wrapper' title='{title}'>{badge}</span>").format(
            title=html.escape(help_text), badge=badge
        )
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
