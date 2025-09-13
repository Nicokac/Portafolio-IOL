import logging
import streamlit as st
from application.auth_service import (
    get_auth_provider,
    InvalidCredentialsError,
    NetworkError,
)
from ui.header import render_header
from shared.config import settings


logger = logging.getLogger(__name__)


def render_footer() -> None:
    """Render developer information at the bottom of the login form."""
    st.caption("Desarrollado por Nicolás Kachuk")


def render_login_page() -> None:
    """Display the login form with header and footer."""
    render_header()

    if not settings.tokens_key:
        if settings.allow_plain_tokens:
            st.warning(
                "IOL_TOKENS_KEY no está configurada; los tokens se guardarán sin cifrar."
            )
        else:
            st.error(
                "IOL_TOKENS_KEY no está configurada. La aplicación no puede continuar."
            )
            return

    err = st.session_state.pop("login_error", "")
    if err:
        st.error(err)

    with st.form("login_form"):
        st.text_input("Usuario", key="IOL_USERNAME")
        st.text_input("Contraseña", type="password", key="IOL_PASSWORD_WIDGET")
        submitted = st.form_submit_button("Iniciar sesión")
        render_footer()

    if submitted:
        user = st.session_state.get("IOL_USERNAME", "")
        password = st.session_state.get("IOL_PASSWORD_WIDGET", "")
        provider = get_auth_provider()
        try:
            provider.login(user, password)
            st.session_state["authenticated"] = True
            st.session_state.pop("force_login", None)
            st.session_state.pop("login_error", None)
            st.session_state.pop("IOL_PASSWORD_WIDGET", None)
            st.rerun()
        except InvalidCredentialsError:
            logger.warning("Fallo de autenticación")
            st.session_state["login_error"] = "Usuario o contraseña inválidos"
            st.session_state.pop("IOL_PASSWORD_WIDGET", None)
            st.rerun()
        except NetworkError:
            logger.warning("Error de conexión durante el login")
            st.session_state["login_error"] = "Error de conexión"
            st.session_state.pop("IOL_PASSWORD_WIDGET", None)
            st.rerun()
        except RuntimeError as e:
            logger.exception("Error durante el login: %s", e)
            st.session_state["login_error"] = str(e)
            st.session_state.pop("IOL_PASSWORD_WIDGET", None)
            st.rerun()
        except Exception:
            logger.exception("Error inesperado durante el login")
            st.session_state["login_error"] = "Error inesperado, contacte soporte"
            st.session_state.pop("IOL_PASSWORD_WIDGET", None)
            st.rerun()
