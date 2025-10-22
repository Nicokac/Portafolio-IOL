"""Controller for authentication orchestration."""

from __future__ import annotations

from infrastructure.iol.client import IIOLProvider
from services.auth_client import build_client, get_auth_provider
from shared.qa_profiler import track_auth_latency
from ui.adapters import auth_ui

LOGIN_AUTH_TIMESTAMP_KEY = "_login_authenticated_at"


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
    return result.client


__all__ = ["build_iol_client", "LOGIN_AUTH_TIMESTAMP_KEY"]
