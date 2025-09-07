from __future__ import annotations
import time
import streamlit as st
from infrastructure.iol.auth import IOLAuth
from shared.config import settings


def render_action_menu() -> None:
    """Render refresh and logout actions in a compact popover."""
    user = st.session_state.get("IOL_USERNAME") or settings.IOL_USERNAME
    password = st.session_state.get("IOL_PASSWORD") or settings.IOL_PASSWORD

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
        err = ""
        with st.spinner("Cerrando sesión..."):
            try:
                IOLAuth(user or "", password or "").clear_tokens()
            except Exception as e:
                err = str(e)
        st.session_state.clear()
        if err:
            st.session_state["logout_error"] = err
        else:
            st.session_state["logout_done"] = True
        st.rerun()

    if st.session_state.pop("show_refresh_toast", False):
        st.toast("Datos actualizados", icon="✅")

    if st.session_state.pop("logout_done", False):
        st.success("Sesión cerrada")
        st.stop()

    err = st.session_state.pop("logout_error", "")
    if err:
        st.error(f"No se pudo cerrar sesión: {err}")
        st.stop()

