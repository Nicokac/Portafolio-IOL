"""Controller for authentication orchestration."""

from __future__ import annotations

import logging

import streamlit as st

from infrastructure.iol.client import IOLClient, IIOLProvider
from services.auth_client import build_client, get_auth_provider
from services.profile_service import fetch_profile
from shared.qa_profiler import track_auth_latency
from ui.adapters import auth_ui

LOGIN_AUTH_TIMESTAMP_KEY = "_login_authenticated_at"


logger = logging.getLogger(__name__)


def build_iol_client() -> IIOLProvider | None:
    """Coordinate between the auth client service and the UI adapter."""

    session_user = auth_ui.get_session_username()
    provider = get_auth_provider()
    with track_auth_latency():
        result = build_client(session_user, provider=provider)

    if result.error:
        auth_ui.set_login_error(result.error_message)
        if result.should_force_login:
            auth_ui.set_force_login(True)
        auth_ui.rerun()
        return None

    auth_ui.mark_authenticated()
    auth_ui.record_auth_timestamp(LOGIN_AUTH_TIMESTAMP_KEY)
    auth_ui.set_login_error(None)

    client = result.client

    if isinstance(client, IOLClient):
        should_refresh = bool(st.session_state.get("_profile_refresh_pending", True))
        if should_refresh:
            try:
                profile_payload = fetch_profile(client)
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug("No se pudo cargar el perfil del inversor", exc_info=True)
                profile_payload = None
            st.session_state["iol_user_profile"] = profile_payload
            st.session_state["_profile_refresh_pending"] = False
    else:
        st.session_state.pop("iol_user_profile", None)

    return client


__all__ = ["build_iol_client", "LOGIN_AUTH_TIMESTAMP_KEY"]
