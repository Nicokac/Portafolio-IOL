"""Funciones de apoyo para el flujo de login."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, MutableMapping, Literal

from shared.config import settings


Severity = Literal["warning", "error"]


@dataclass(frozen=True)
class TokenKeyValidation:
    """Resultado de validar la clave de tokens."""

    can_proceed: bool
    level: Severity | None = None
    message: str | None = None


def validate_tokens_key() -> TokenKeyValidation:
    """Verifica si la aplicación puede continuar con el proceso de login.

    Si ``IOL_TOKENS_KEY`` no está configurada y no se permite guardar
    tokens en texto plano, el login no debería continuar.
    """

    allow_plain_tokens = getattr(settings, "allow_plain_tokens", False)
    env_flag = os.getenv("IOL_ALLOW_PLAIN_TOKENS")
    if env_flag is not None:
        allow_plain_tokens = str(env_flag).lower() in {"1", "true", "yes", "on"}

    if settings.tokens_key:
        return TokenKeyValidation(can_proceed=True)

    if allow_plain_tokens:
        return TokenKeyValidation(
            can_proceed=True,
            level="warning",
            message="IOL_TOKENS_KEY no está configurada; los tokens se guardarán sin cifrar.",
        )

    return TokenKeyValidation(
        can_proceed=False,
        level="error",
        message="IOL_TOKENS_KEY no está configurada. La aplicación no puede continuar.",
    )


def clear_password_keys(state: MutableMapping[str, Any]) -> None:
    """Elimina cualquier clave relacionada a contraseñas del estado dado."""

    for key in list(state.keys()):
        if "password" in str(key).lower():
            state.pop(key, None)
