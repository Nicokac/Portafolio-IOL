"""Jerarquía de errores de aplicación compartida."""

from __future__ import annotations


class AppError(Exception):
    """Excepción base para errores específicos de la aplicación."""


class NetworkError(AppError):
    """Representa fallas relacionadas con la red."""


class InvalidCredentialsError(AppError):
    """Se genera cuando las credenciales proporcionadas no son válidas."""


class TimeoutError(NetworkError):
    """Se genera cuando se agota el tiempo de espera de una operación de red."""


class ExternalAPIError(AppError):
    """Se genera cuando una API externa devuelve un error."""


class CacheUnavailableError(AppError):
    """Se genera cuando el backend de caché no está disponible."""


__all__ = [
    "AppError",
    "NetworkError",
    "InvalidCredentialsError",
    "TimeoutError",
    "ExternalAPIError",
    "CacheUnavailableError",
]
