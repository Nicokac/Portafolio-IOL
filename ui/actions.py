from __future__ import annotations
import time
import streamlit as st
from infrastructure.iol.auth import IOLAuth
from shared.config import settings
from infrastructure.cache import cache


def render_action_menu() -> None:
    """Render refresh and logout actions in a compact popover."""
    # user = cache.session_state.get("IOL_USERNAME") or settings.IOL_USERNAME
    # password = cache.session_state.get("IOL_PASSWORD") or settings.IOL_PASSWORD
    if cache.session_state.get("force_login"):
        user = cache.session_state.get("IOL_USERNAME")
        password = cache.session_state.get("IOL_PASSWORD")
    else:
        user = cache.session_state.get("IOL_USERNAME") or settings.IOL_USERNAME
        password = cache.session_state.get("IOL_PASSWORD") or settings.IOL_PASSWORD

    pop = st.popover("⚙️ Acciones")
    with pop:
        st.caption("Operaciones rápidas")
        c1, c2 = st.columns(2)
        if c1.button("⟳ Refrescar", use_container_width=True):
            cache.session_state["refresh_pending"] = True
            st.rerun()
        if c2.button("🔒 Cerrar sesión", use_container_width=True):
            cache.session_state["logout_pending"] = True
            st.rerun()

    if cache.session_state.pop("refresh_pending", False):
        with st.spinner("Actualizando datos..."):
            cache.session_state["last_refresh"] = time.time()
        cache.session_state["show_refresh_toast"] = True
        st.rerun()

    if cache.session_state.pop("logout_pending", False):
        err = ""
        with st.spinner("Cerrando sesión..."):
            try:
                IOLAuth(user or "", password or "").clear_tokens()
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("Error al limpiar tokens: %s", e)
                err = str(e)
        cache.session_state.clear()
        cache.session_state["force_login"] = True
        if err:
            cache.session_state["logout_error"] = err
        else:
            cache.session_state["logout_done"] = True
        st.rerun()

    if cache.session_state.pop("show_refresh_toast", False):
        st.toast("Datos actualizados", icon="✅")

    if cache.session_state.pop("logout_done", False):
        st.success("Sesión cerrada")
        st.stop()

    err = cache.session_state.pop("logout_error", "")
    if err:
        st.error(f"No se pudo cerrar sesión: {err}")
        st.stop()

