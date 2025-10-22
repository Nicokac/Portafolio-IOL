"""Domain service for obtaining authenticated IOL clients."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from application.auth_service import get_auth_provider as _get_auth_provider
from infrastructure.iol.auth import InvalidCredentialsError
from infrastructure.iol.client import IIOLProvider
from shared.fragment_state import prepare_persistent_fragment_restore
from shared.user_actions import log_user_action
from shared.visual_cache_prewarm import prewarm_visual_cache

logger = logging.getLogger(__name__)


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


@dataclass(slots=True)
class AuthClientResult:
    """Outcome of attempting to build an authenticated IOL client."""

    client: IIOLProvider | None
    error: Exception | None
    error_message: str | None
    should_force_login: bool
    telemetry: dict[str, Any]


def get_auth_provider():
    """Return the configured authentication provider."""

    return _get_auth_provider()


def build_client(
    session_user: str | None,
    provider: Any | None = None,
) -> AuthClientResult:
    """Build an authenticated client without touching any UI concerns."""

    provider = provider or get_auth_provider()
    telemetry: dict[str, Any] = {
        "provider": provider.__class__.__name__,
        "status": "success",
    }
    masked_user = _mask_username(session_user)
    if session_user:
        telemetry["user"] = masked_user

    with _get_performance_timer()("token_refresh", extra=telemetry):
        try:
            client, error = provider.build_client()
        except Exception:
            telemetry["status"] = "error"
            raise
        if error is not None:
            telemetry["status"] = "error"

    if error:
        logger.exception("build_iol_client failed", exc_info=error)
        if isinstance(error, InvalidCredentialsError):
            message = "Credenciales inválidas"
            log_user_action(
                "session_timeout",
                {
                    "provider": telemetry["provider"],
                    "user": masked_user,
                },
            )
        else:
            message = "Error de conexión"
            log_user_action(
                "login_error",
                {
                    "provider": telemetry["provider"],
                    "user": masked_user,
                },
            )
        return AuthClientResult(
            client=None,
            error=error,
            error_message=message,
            should_force_login=True,
            telemetry=telemetry,
        )

    try:
        prewarm_visual_cache()
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo precalentar la caché visual tras el login", exc_info=True)

    log_user_action(
        "login_success",
        {
            "provider": telemetry["provider"],
            "user": masked_user,
        },
    )

    try:
        prepare_persistent_fragment_restore()
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug(
            "No se pudo preparar la restauración persistente de fragmentos",
            exc_info=True,
        )

    return AuthClientResult(
        client=client,
        error=None,
        error_message=None,
        should_force_login=False,
        telemetry=telemetry,
    )


__all__ = ["AuthClientResult", "build_client", "get_auth_provider"]
