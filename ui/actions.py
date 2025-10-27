from __future__ import annotations

import logging

import streamlit as st

from application import auth_service
from shared.debug.rerun_trace import mark_event, safe_rerun, safe_stop
from shared.errors import AppError

logger = logging.getLogger(__name__)


def render_action_menu(container=None, *, show_refresh: bool = True) -> None:
    """Render refresh and logout actions in a compact control panel."""

    host = container if container is not None else st.container(border=True)
    with host:
        st.markdown(
            "<div class='control-panel__section control-panel__actions'>",
            unsafe_allow_html=True,
        )
        if show_refresh:
            st.markdown("### ⚙️ Acciones rápidas")
            st.caption("Mantén esta sección a la vista para actuar sin perder contexto.")
            refresh_col, logout_col = st.columns(2)
        else:
            st.markdown("### 🔐 Sesión")
            st.caption("Consultá el estado actual y cerrá sesión cuando lo necesites.")
            (logout_col,) = st.columns(1)
            refresh_col = None
        if show_refresh and refresh_col and refresh_col.button("⟳ Refrescar", width="stretch", type="secondary"):
            st.session_state["refresh_pending"] = True
            st.session_state["ui_refresh_silent"] = True
        if logout_col.button(
            "🔒 Cerrar sesión",
            width="stretch",
            help="Cierra inmediatamente tu sesión actual",
            type="secondary",
        ):
            st.session_state["logout_pending"] = True
            mark_event("rerun", "logout_requested")
            safe_rerun("logout_requested")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("logout_pending", False):
        st.markdown(
            "<div class='control-panel__banner control-panel__banner--logout fade-out'>"
            "Iniciando cierre de sesión..."
            "</div>",
            unsafe_allow_html=True,
        )
        st.session_state.pop("logout_pending", None)
        try:
            with st.spinner("Cerrando sesión..."):
                auth_service.logout(st.session_state.get("IOL_USERNAME", ""))
        except AppError as err:
            st.error(str(err))
            mark_event("stop", "logout_app_error")
            safe_stop("logout_app_error")
        except Exception:
            logger.exception("Error inesperado al cerrar sesión")
            st.error("No se pudo cerrar sesión, intente nuevamente más tarde")
            mark_event("stop", "logout_exception")
            safe_stop("logout_exception")

    if st.session_state.pop("logout_done", False):
        st.success("Sesión cerrada")
        mark_event("stop", "logout_done")
        safe_stop("logout_done")

    err = st.session_state.pop("logout_error", "")
    if err:
        st.error(f"No se pudo cerrar sesión: {err}")
        mark_event("stop", "logout_error")
        safe_stop("logout_error")
