import logging
import streamlit as st
from application import auth_service
from application.auth_service import AuthenticationError
from ui.header import render_header
from infrastructure.cache import cache


logger = logging.getLogger(__name__)


def render_footer() -> None:
    """Render developer information at the bottom of the login form."""
    st.caption("Desarrollado por Nicolás Kachuk")


def render_login_page() -> None:
    """Display the login form with header and footer."""
    render_header()

    err = cache.session_state.pop("login_error", "")
    if err:
        st.error(err)

    with st.form("login_form"):
        st.text_input("Usuario", key="IOL_USERNAME")
        st.text_input("Contraseña", type="password", key="IOL_PASSWORD")
        submitted = st.form_submit_button("Iniciar sesión")
        render_footer()

    if submitted:
        user = cache.session_state.get("IOL_USERNAME", "")
        password = cache.session_state.get("IOL_PASSWORD", "")
        try:
            auth_service.login(user, password)
            cache.session_state.pop("force_login", None)
            cache.session_state.pop("login_error", None)
            st.rerun()
        except AuthenticationError:
            logger.exception("Fallo de autenticación")
            cache.session_state["login_error"] = "Usuario o contraseña inválidos"
            cache.session_state["IOL_PASSWORD"] = ""
            st.rerun()
        except Exception:
            logger.exception("Error inesperado durante el login")
            cache.session_state["login_error"] = "Error inesperado, contacte soporte"
            cache.session_state["IOL_PASSWORD"] = ""
            st.rerun()
