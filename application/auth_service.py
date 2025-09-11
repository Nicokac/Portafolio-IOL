from __future__ import annotations

"""Servicios de autenticación para la aplicación."""

from infrastructure.iol.auth import IOLAuth


def login(user: str, password: str):
    """Autentica al usuario contra IOL.

    Devuelve el diccionario de tokens si el login es exitoso.
    Lanza ``RuntimeError`` si no se obtuvo un ``access_token`` válido.
    """
    tokens = IOLAuth(user, password).login()
    if not tokens.get("access_token"):
        raise RuntimeError("Credenciales inválidas")
    return tokens


def logout(user: str, password: str = "") -> None:
    """Elimina los tokens de autenticación para forzar un nuevo login."""
    IOLAuth(user, password).clear_tokens()
