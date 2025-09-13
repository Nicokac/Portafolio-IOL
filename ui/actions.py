from __future__ import annotations
import time
import streamlit as st
from application import auth_service


def render_action_menu() -> None:
    """Render refresh and logout actions in a compact popover."""

    pop = st.popover("⚙️ Acciones")
    with pop:
        st.caption("Operaciones rápidas")
        c1, c2 = st.columns(2)
        if c1.button("⟳ Refrescar", use_container_width=True):
            st.session_state["refresh_pending"] = True
            st.rerun()
        if c2.button("🔒 Cerrar sesión", use_container_width=True):
            st.session_state["logout_pending"] = True
            st.rerun()

    if st.session_state.pop("refresh_pending", False):
        with st.spinner("Actualizando datos..."):
            st.session_state["last_refresh"] = time.time()
        st.session_state["show_refresh_toast"] = True
        st.rerun()

    if st.session_state.pop("logout_pending", False):
        with st.spinner("Cerrando sesión..."):
            auth_service.logout()

    if st.session_state.pop("show_refresh_toast", False):
        st.toast("Datos actualizados", icon="✅")

    if st.session_state.pop("logout_done", False):
        st.success("Sesión cerrada")
        st.stop()

    err = st.session_state.pop("logout_error", "")
    if err:
        st.error(f"No se pudo cerrar sesión: {err}")
        st.stop()

