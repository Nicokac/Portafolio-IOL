import importlib
import logging
import os
from functools import lru_cache

import streamlit as st

from application.auth_service import get_auth_provider
from application.login_service import clear_password_keys, validate_tokens_key
from services.auth import MAX_TOKEN_TTL_SECONDS, describe_active_token, generate_token
from shared.config import settings  # Re-exported for backwards compatibility
from shared.errors import AppError, InvalidCredentialsError, NetworkError
from shared.time_provider import TimeProvider


logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _lazy_module(name: str):
    return importlib.import_module(name)


@lru_cache(maxsize=None)
def _lazy_attr(module: str, attr: str):
    return getattr(_lazy_module(module), attr)


def _get_auth_token_ttl() -> int:
    """Return the configured TTL for API tokens capped at fifteen minutes."""

    raw_value = os.environ.get("FASTAPI_AUTH_TTL", "900")
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return 900
    return max(1, min(value, MAX_TOKEN_TTL_SECONDS))


def render_login_page() -> None:
    """Display the login form with header and footer."""
    render_header = _lazy_attr("ui.header", "render_header")
    render_footer = _lazy_attr("ui.footer", "render_footer")

    render_header()

    validation = validate_tokens_key()
    if validation.message:
        if validation.level == "warning":
            st.warning(validation.message)
        elif validation.level == "error":
            st.error(validation.message)
    if not validation.can_proceed:
        render_footer()
        return

    err = st.session_state.pop("login_error", "")
    if err:
        st.error(err)

    with st.form("login_form", clear_on_submit=False):
        raw_user = st.text_input("Usuario", key="login_username")
        raw_password = st.text_input(
            "Contraseña",
            type="password",
            key="login_password",
        )
        submitted = st.form_submit_button("Iniciar sesión")

    user = (raw_user or "").strip()
    password = raw_password or ""

    render_footer()

    if submitted:
        provider = get_auth_provider()

        if not user or not password:
            logger.debug(
                "Login submit ignored due to missing credentials (user_present=%s)",
                bool(user),
            )
            st.session_state["login_error"] = "Ingresá usuario y contraseña para continuar"
            clear_password_keys(st.session_state)
            st.rerun()
            return

        st.session_state["IOL_USERNAME"] = user
        st.session_state.pop("cli", None)

        try:
            provider.login(user, password)
            token = generate_token(
                user,
                _get_auth_token_ttl(),
            )
            st.session_state["auth_token"] = token
            snapshot = describe_active_token(token)
            if snapshot and isinstance(snapshot.get("claims"), dict):
                st.session_state["auth_token_claims"] = dict(snapshot["claims"])
            else:
                st.session_state.pop("auth_token_claims", None)
            st.session_state["auth_token_refreshed_at"] = TimeProvider.now()
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
        except AppError as err:
            logger.warning("Error controlado durante el login: %s", err)
            msg = str(err)
            st.error(msg)
            st.session_state["login_error"] = msg
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
