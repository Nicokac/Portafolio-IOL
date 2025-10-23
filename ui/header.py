import logging
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from shared.cache import cache
from shared.utils import _as_float_or_none
from ui.palette import get_active_palette

logger = logging.getLogger(__name__)

_NOTIFICATION_DATA_KEY = "_header_notification_data"
_NOTIFICATION_MARKER_KEY = "_header_notification_marker"
_NOTIFICATION_LAST_MESSAGE_KEY = "_header_notification_message"
_NOTIFICATION_PLACEHOLDER_KEY = "_header_notification_placeholder"
_NOTIFICATION_VARIANT_KEY = "_header_notification_variant"

_SKIP_NOTIFICATION_FETCH = object()

_PORTFOLIO_FX_LABEL_KEY = "portfolio_summary_fx_label"
_PORTFOLIO_FX_VALUE_KEY = "portfolio_summary_fx_value"


def _current_notification_marker() -> Any:
    refreshed = st.session_state.get("auth_token_refreshed_at")
    if refreshed:
        return refreshed
    return st.session_state.get("cache_key")


def _fetch_notification() -> Optional[dict[str, Any]]:
    if not _can_attempt_notification_fetch():
        logger.debug("Se omite fetch de notificaciones: sesión no autenticada o sin tokens disponibles")
        return _SKIP_NOTIFICATION_FETCH

    try:
        from services.cache import build_iol_client as _build_iol_client
    except Exception:  # pragma: no cover - defensive import
        logger.debug("No se pudo importar build_iol_client para notificaciones", exc_info=True)
        return None

    try:
        client, error = _build_iol_client()
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("No se pudo inicializar IOLClient para notificaciones: %s", exc, exc_info=True)
        return None

    if error is not None or client is None:
        if error is not None:
            logger.debug("build_iol_client devolvió error al obtener la notificación: %s", error)
        return None

    getter = getattr(client, "get_notification", None)
    if not callable(getter):
        return None

    try:
        return getter()
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("No se pudo obtener la notificación activa: %s", exc)
        return None


def _load_notification() -> Optional[dict[str, Any]]:
    marker = _current_notification_marker()
    cached_marker = st.session_state.get(_NOTIFICATION_MARKER_KEY)
    if cached_marker == marker and _NOTIFICATION_DATA_KEY in st.session_state:
        return st.session_state.get(_NOTIFICATION_DATA_KEY)

    notification = _fetch_notification()
    if notification is _SKIP_NOTIFICATION_FETCH:
        return st.session_state.get(_NOTIFICATION_DATA_KEY)

    st.session_state[_NOTIFICATION_MARKER_KEY] = marker
    st.session_state[_NOTIFICATION_DATA_KEY] = notification
    return notification


def _can_attempt_notification_fetch() -> bool:
    state = st.session_state
    if state.get("force_login"):
        return False
    if state.get("auth_token_refreshed_at") or state.get("authenticated"):
        return True

    try:
        tokens_file = cache.get("tokens_file")
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("No se pudo acceder al cache de tokens para notificaciones", exc_info=True)
        return False

    if not tokens_file:
        return False

    try:
        path = Path(tokens_file)
    except (TypeError, ValueError, OSError):  # pragma: no cover - defensive guard
        logger.debug("Ruta de tokens inválida para notificaciones: %s", tokens_file, exc_info=True)
        return False

    return path.exists()


def _should_use_warning(message: str) -> bool:
    lowered = message.lower()
    return any(keyword in lowered for keyword in ("mantenimiento", "maintenance", "alerta"))


def _render_notification_banner() -> None:
    notification = _load_notification()
    if not notification:
        st.session_state.pop(_NOTIFICATION_LAST_MESSAGE_KEY, None)
        st.session_state.pop(_NOTIFICATION_VARIANT_KEY, None)
        placeholder = st.session_state.pop(_NOTIFICATION_PLACEHOLDER_KEY, None)
        if placeholder is not None:
            try:
                placeholder.empty()
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("No se pudo limpiar el contenedor de notificaciones", exc_info=True)
        return

    message = notification.get("mensaje") if isinstance(notification, dict) else None
    if not isinstance(message, str):
        return
    message = message.strip()
    if not message:
        return

    placeholder = st.session_state.get(_NOTIFICATION_PLACEHOLDER_KEY)
    if placeholder is None:
        placeholder = st.empty()
        st.session_state[_NOTIFICATION_PLACEHOLDER_KEY] = placeholder

    variant = "warning" if _should_use_warning(message) else "info"
    last_message = st.session_state.get(_NOTIFICATION_LAST_MESSAGE_KEY)
    last_variant = st.session_state.get(_NOTIFICATION_VARIANT_KEY)
    if last_message != message or last_variant != variant:
        st.session_state[_NOTIFICATION_LAST_MESSAGE_KEY] = message
        st.session_state[_NOTIFICATION_VARIANT_KEY] = variant

    if variant == "warning":
        placeholder.warning(message)
    else:
        placeholder.info(message)


def render_header(rates=None):
    """Render the application header with contextual actions."""

    _render_notification_banner()

    pal = get_active_palette()

    st.markdown(
        """
        <div style="margin:0 auto; padding:1rem 0; text-align:center;">
            <div style="display:inline-flex; gap:0.8rem; align-items:flex-start; justify-content:center;">
                <span style="font-size:2.2rem; line-height:1;">📈</span>
                <div>
                    <h1 style="margin:0; font-size:1.8rem;">IOL — Portafolio en vivo</h1>
                    <p style="margin:0.4rem 0 0; color:#555;">
                        Acceso operativo en modo <strong>solo lectura</strong>
                        para monitorear posiciones y cotizaciones.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if rates:
        render_fx_summary_in_header(rates, palette=pal)


def render_fx_summary_in_header(rates: dict, palette=None):
    """Render a summary of foreign exchange rates in the header.

    Parameters
    ----------
    rates : dict
        Diccionario con las cotizaciones a mostrar.
    """
    if not rates:
        return

    pal = palette or get_active_palette()

    desired = [
        ("💵 Oficial", "oficial"),
        ("📈 MEP", "mep"),
    ]
    items = [(label, _as_float_or_none(rates.get(key))) for label, key in desired]

    cols = st.columns(len(items))
    active_label = str(st.session_state.get(_PORTFOLIO_FX_LABEL_KEY, "") or "").lower()
    active_value = _as_float_or_none(st.session_state.get(_PORTFOLIO_FX_VALUE_KEY))

    neutral_bg = "rgba(0,0,0,0.04)" if pal.bg == "#FFFFFF" else "rgba(255,255,255,0.08)"
    neutral_border = "1px solid rgba(0,0,0,0.05)" if pal.bg == "#FFFFFF" else "1px solid rgba(255,255,255,0.15)"

    for col, (label, value) in zip(cols, items):
        label_text = label.split(" ", 1)[1] if " " in label else label
        highlight = label_text.lower() == active_label
        if (
            not highlight
            and active_value is not None
            and value is not None
        ):
            diff = abs(float(value) - float(active_value))
            tolerance = max(2.5, float(active_value) * 0.01)
            highlight = diff <= tolerance

        bg_color = pal.highlight_bg if highlight else neutral_bg
        text_color = pal.highlight_text if highlight else pal.text
        border = f"2px solid {pal.accent}" if highlight else neutral_border
        sublabel = "Usado en totales" if highlight else "Referencia"
        shadow = f"box-shadow:0 6px 18px {pal.accent}33;" if highlight else ""
        value_str = (
            f"$ {value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
            if isinstance(value, (int, float)) and value is not None
            else "–"
        )
        card_style = (
            "display:flex; flex-direction:column; gap:0.3rem; "
            "padding:0.9rem 1.1rem; border-radius:0.9rem; text-align:center; "
            f"background-color:{bg_color}; color:{text_color}; border:{border}; {shadow}"
        )
        col.markdown(
            f"""
            <div style="{card_style}">
                <span style="font-size:0.82rem; letter-spacing:0.03em; text-transform:uppercase; opacity:0.85;">{label}</span>
                <span style="font-size:1.45rem; font-weight:700; line-height:1;">{value_str}</span>
                <span style="font-size:0.75rem; opacity:0.75;">{sublabel}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
