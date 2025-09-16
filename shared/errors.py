"""Excepciones compartidas para la aplicación."""
from __future__ import annotations


class Error(Exception):
    """Excepción base para errores de la aplicación."""


class InvalidCredentialsError(Error):
    """Se lanza cuando el usuario o contraseña son inválidos."""


class NetworkError(Error):
    """Se lanza ante problemas de conectividad con la API."""


class TimeoutError(NetworkError):
    """Se lanza ante timeouts de red."""


__all__ = [
    "Error",
    "InvalidCredentialsError",
    "NetworkError",
    "TimeoutError",
]
