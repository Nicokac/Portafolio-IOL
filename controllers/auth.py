import importlib
import logging
import time
from functools import lru_cache

import streamlit as st

from infrastructure.iol.client import IIOLProvider
from application.auth_service import get_auth_provider
from infrastructure.iol.auth import InvalidCredentialsError

logger = logging.getLogger(__name__)

LOGIN_AUTH_TIMESTAMP_KEY = "_login_authenticated_at"


@lru_cache(maxsize=1)
def _get_performance_timer():
    module = importlib.import_module("services.performance_timer")
    return getattr(module, "performance_timer")


def _mask_username(value: str | None) -> str:
    username = (value or "").strip()
    if not username:
        return "anon"
    if len(username) <= 3:
        return username[:1] + "**"
    return f"{username[:3]}***"


def build_iol_client() -> IIOLProvider | None:
    provider = get_auth_provider()
    telemetry: dict[str, object] = {
        "provider": provider.__class__.__name__,
        "status": "success",
    }
    session_user = st.session_state.get("IOL_USERNAME")
    if session_user:
        telemetry["user"] = _mask_username(str(session_user))
    with _get_performance_timer()("token_refresh", extra=telemetry):
        try:
            cli, error = provider.build_client()
        except Exception:
            telemetry["status"] = "error"
            raise
        if error is not None:
            telemetry["status"] = "error"
    if error:
        logger.exception("build_iol_client failed", exc_info=error)
        if isinstance(error, InvalidCredentialsError):
            st.session_state["login_error"] = "Credenciales inválidas"
        else:
            st.session_state["login_error"] = "Error de conexión"
        st.session_state["force_login"] = True
        st.rerun()
        return None
    st.session_state["authenticated"] = True
    try:
        st.session_state[LOGIN_AUTH_TIMESTAMP_KEY] = time.perf_counter()
    except Exception:  # pragma: no cover - session_state may be read-only in tests
        logger.debug("No se pudo registrar el timestamp de autenticación", exc_info=True)
    return cli


__all__ = ["build_iol_client", "LOGIN_AUTH_TIMESTAMP_KEY"]

