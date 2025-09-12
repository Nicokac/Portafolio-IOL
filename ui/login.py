import logging
import streamlit as st
from application.auth_service import AuthenticationError, get_auth_provider
from ui.header import render_header


logger = logging.getLogger(__name__)


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
        provider = get_auth_provider()
        try:
            provider.login(user, password)
            st.session_state.pop("force_login", None)
            st.session_state.pop("login_error", None)
            st.rerun()
        except AuthenticationError:
            logger.warning("Fallo de autenticación")
            logger.debug("Error de autenticación")
            st.session_state["login_error"] = "Usuario o contraseña inválidos"
            st.session_state["IOL_PASSWORD"] = ""
            st.rerun()
        except Exception:
            logger.exception("Error inesperado durante el login")
            st.session_state["login_error"] = "Error inesperado, contacte soporte"
            st.session_state["IOL_PASSWORD"] = ""
            st.rerun()
