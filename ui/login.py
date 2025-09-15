import logging
import streamlit as st
from application.auth_service import (
    get_auth_provider,
    InvalidCredentialsError,
    NetworkError,
)
from application.login_service import clear_password_keys, validate_tokens_key
from ui.footer import render_footer
from ui.header import render_header
from shared.config import settings  # Re-exported for backwards compatibility


logger = logging.getLogger(__name__)


def render_login_page() -> None:
    """Display the login form with header and footer."""
    render_header()

    validation = validate_tokens_key()
    if validation.message:
        if validation.level == "warning":
            st.warning(validation.message)
        elif validation.level == "error":
            st.error(validation.message)
    if not validation.can_proceed:
        return

    err = st.session_state.pop("login_error", "")
    if err:
        st.error(err)

    with st.form("login_form"):
        user = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        submitted = st.form_submit_button("Iniciar sesión")
        render_footer()

    if submitted:
        provider = get_auth_provider()

        try:
            provider.login(user, password)
            st.session_state["authenticated"] = True
            st.session_state.pop("force_login", None)
            st.session_state.pop("login_error", None)
            clear_password_keys(st.session_state)
            st.rerun()
        except InvalidCredentialsError:
            logger.warning("Fallo de autenticación")
            st.session_state["login_error"] = "Usuario o contraseña inválidos"
            clear_password_keys(st.session_state)
            st.rerun()
        except NetworkError:
            logger.warning("Error de conexión durante el login")
            st.session_state["login_error"] = "Error de conexión"
            clear_password_keys(st.session_state)
            st.rerun()
        except RuntimeError as e:
            logger.exception("Error durante el login: %s", e)
            st.session_state["login_error"] = str(e)
            clear_password_keys(st.session_state)
            st.rerun()
        except Exception:
            logger.exception("Error inesperado durante el login")
            st.session_state["login_error"] = "Error inesperado, contacte soporte"
            clear_password_keys(st.session_state)
            st.rerun()


__all__ = ["render_login_page", "settings"]
