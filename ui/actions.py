from __future__ import annotations
import time
import streamlit as st
from infrastructure.iol.auth import IOLAuth

def render_action_menu(user: str, password: str) -> None:
    """Render refresh and relogin actions in a compact popover."""
    pop = st.popover("⚙️ Acciones")
    with pop:
        st.caption("Operaciones rápidas")
        c1, c2 = st.columns(2)
        if c1.button("⟳ Refrescar", use_container_width=True):
            st.session_state["refresh_pending"] = True
            st.rerun()
        if c2.button("🔄 Relogin", use_container_width=True):
            st.session_state["relogin_pending"] = True
            st.rerun()

    if st.session_state.pop("refresh_pending", False):
        with st.spinner("Actualizando datos..."):
            st.session_state["last_refresh"] = time.time()
        st.session_state["show_refresh_toast"] = True
        st.rerun()

    if st.session_state.pop("relogin_pending", False):
        with st.spinner("Eliminando tokens..."):
            try:
                IOLAuth(user, password).clear_tokens()
                st.session_state["client_salt"] = int(time.time())
                st.session_state["relogin_done"] = True
            except Exception as e:
                st.session_state["relogin_error"] = str(e)
        st.rerun()

    if st.session_state.pop("show_refresh_toast", False):
        st.toast("Datos actualizados", icon="✅")

    if st.session_state.pop("relogin_done", False):
        st.success("Tokens eliminados. Recargando…")

    err = st.session_state.pop("relogin_error", "")
    if err:
        st.error(f"No se pudo limpiar tokens: {err}")
        st.stop()