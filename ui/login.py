import streamlit as st
from application import auth_service
from ui.header import render_header


def render_footer() -> None:
    """Render developer information at the bottom of the login form."""
    st.caption("Desarrollado por Nicolás Kachuk")


def render_login_page() -> None:
    """Display the login form with header and footer."""
    render_header()

    err = st.session_state.pop("login_error", "")
    if err:
        st.error(err)

    with st.form("login_form"):
        st.text_input("Usuario", key="IOL_USERNAME")
        st.text_input("Contraseña", type="password", key="IOL_PASSWORD")
        submitted = st.form_submit_button("Iniciar sesión")
        render_footer()

    if submitted:
        user = st.session_state.get("IOL_USERNAME", "")
        password = st.session_state.get("IOL_PASSWORD", "")
        try:
            auth_service.login(user, password)
            st.session_state.pop("force_login", None)
            st.session_state.pop("login_error", None)
            st.rerun()
        except Exception as e:
            st.session_state["login_error"] = str(e)
            st.session_state["IOL_PASSWORD"] = ""
            st.rerun()
