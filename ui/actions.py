from __future__ import annotations

import logging
import time
import streamlit as st

from application import auth_service
from shared.errors import AppError


logger = logging.getLogger(__name__)


def render_action_menu(container=None) -> None:
    """Render refresh and logout actions in a compact control panel."""

    host = container if container is not None else st.container(border=True)
    with host:
        st.markdown(
            "<div class='control-panel__section control-panel__actions'>",
            unsafe_allow_html=True,
        )
        st.markdown("### ⚙️ Acciones rápidas")
        st.caption("Mantén esta sección a la vista para actuar sin perder contexto.")
        c1, c2 = st.columns(2)
        if c1.button("⟳ Refrescar", width="stretch", type="secondary"):
            st.session_state["refresh_pending"] = True
            st.rerun()
        if c2.button(
            "🔒 Cerrar sesión",
            width="stretch",
            help="Cierra inmediatamente tu sesión actual",
            type="secondary",
        ):
            st.session_state["logout_pending"] = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.pop("refresh_pending", False):
        with st.spinner("Actualizando datos..."):
            st.session_state["last_refresh"] = time.time()
        st.session_state["show_refresh_toast"] = True
        st.rerun()

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
            st.stop()
        except Exception:
            logger.exception("Error inesperado al cerrar sesión")
            st.error("No se pudo cerrar sesión, intente nuevamente más tarde")
            st.stop()

    if st.session_state.pop("show_refresh_toast", False):
        st.toast("Datos actualizados", icon="✅")

    if st.session_state.pop("logout_done", False):
        st.success("Sesión cerrada")
        st.stop()

    err = st.session_state.pop("logout_error", "")
    if err:
        st.error(f"No se pudo cerrar sesión: {err}")
        st.stop()

