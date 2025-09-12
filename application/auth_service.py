from __future__ import annotations

"""Servicios de autenticación para la aplicación."""

from pathlib import Path

from infrastructure.iol.auth import IOLAuth
from shared.cache import cache


class AuthenticationError(Exception):
    """Se lanza cuando la autenticación contra IOL falla."""


def login(user: str, password: str):
    """Autentica al usuario contra IOL.

    Devuelve el diccionario de tokens si el login es exitoso.
    Lanza ``AuthenticationError`` si no se obtuvo un ``access_token`` válido.
    """
    tokens_path = Path("tokens") / f"{user}.json"
    cache.set("tokens_file", str(tokens_path))
    tokens = IOLAuth(user, password, tokens_file=tokens_path).login()
    if not tokens.get("access_token"):
        raise AuthenticationError("Credenciales inválidas")
    return tokens


def logout(user: str, password: str = "") -> None:
    """Elimina los tokens de autenticación para forzar un nuevo login."""
    tokens_path = Path("tokens") / f"{user}.json"
    IOLAuth(user, password, tokens_file=tokens_path).clear_tokens()
    cache.pop("tokens_file", None)
