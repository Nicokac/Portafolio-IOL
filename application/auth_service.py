from __future__ import annotations

"""Servicios de autenticación para la aplicación."""

from infrastructure.iol.auth import IOLAuth


class AuthenticationError(Exception):
    """Se lanza cuando la autenticación contra IOL falla."""


def login(user: str, password: str):
    """Autentica al usuario contra IOL.

    Devuelve el diccionario de tokens si el login es exitoso.
    Lanza ``AuthenticationError`` si no se obtuvo un ``access_token`` válido.
    """
    tokens = IOLAuth(user, password).login()
    if not tokens.get("access_token"):
        raise AuthenticationError("Credenciales inválidas")
    return tokens


def logout(user: str, password: str = "") -> None:
    """Elimina los tokens de autenticación para forzar un nuevo login."""
    IOLAuth(user, password).clear_tokens()
