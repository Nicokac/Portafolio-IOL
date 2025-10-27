"""Adapter that encapsulates Streamlit interactions for authentication."""

from __future__ import annotations

import logging
import time

import streamlit as st

from shared.debug.rerun_trace import mark_event, safe_rerun

logger = logging.getLogger(__name__)


def get_session_username() -> str | None:
    """Return the username stored in the Streamlit session, if any."""

    value = st.session_state.get("IOL_USERNAME")
    return None if value is None else str(value)


def set_login_error(message: str | None) -> None:
    """Persist the login error message in the session."""

    if message is None:
        st.session_state.pop("login_error", None)
    else:
        st.session_state["login_error"] = message


def set_force_login(value: bool = True) -> None:
    """Toggle the flag that forces a login screen rerender."""

    st.session_state["force_login"] = value


def mark_authenticated() -> None:
    """Mark the current session as authenticated."""

    st.session_state["authenticated"] = True


def record_auth_timestamp(key: str) -> None:
    """Store the authentication timestamp for later diagnostics."""

    try:
        st.session_state[key] = time.perf_counter()
    except Exception:  # pragma: no cover - session_state may be read-only in tests
        logger.debug("No se pudo registrar el timestamp de autenticación", exc_info=True)


def rerun() -> None:
    """Trigger a Streamlit rerun."""

    mark_event("rerun", "auth_adapter")
    safe_rerun("auth_adapter")


__all__ = [
    "get_session_username",
    "mark_authenticated",
    "record_auth_timestamp",
    "rerun",
    "set_force_login",
    "set_login_error",
]
