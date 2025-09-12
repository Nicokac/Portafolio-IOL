from __future__ import annotations

"""Servicios de autenticación para la aplicación."""

from pathlib import Path
from typing import Protocol, Tuple, Any
import re
import hashlib

import streamlit as st

from infrastructure.iol.auth import IOLAuth
from shared.cache import cache
from shared.config import settings


class AuthenticationError(Exception):
    """Se lanza cuando la autenticación falla."""


class AuthenticationProvider(Protocol):
    """Define las operaciones básicas de autenticación."""

    def login(self, user: str, password: str) -> dict:
        """Realiza el proceso de login y devuelve tokens."""

    def logout(self, user: str, password: str = "") -> None:
        """Limpia cualquier recurso asociado a la autenticación."""

    def build_client(self) -> Tuple[Any | None, Exception | None]:
        """Construye el cliente de datos asociado al proveedor."""


class IOLAuthenticationProvider(AuthenticationProvider):
    """Proveedor de autenticación basado en IOL."""

    def login(self, user: str, password: str) -> dict:
        sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", user)
        tokens_path = Path("tokens") / f"{sanitized}.json"
        cache.set("tokens_file", str(tokens_path))
        tokens = IOLAuth(
            user,
            password,
            tokens_file=tokens_path,
            allow_plain_tokens=settings.allow_plain_tokens,
        ).login()
        if not tokens.get("access_token"):
            raise AuthenticationError("Credenciales inválidas")
        return tokens

    def logout(self, user: str, password: str = "") -> None:
        sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", user)
        tokens_path = Path("tokens") / f"{sanitized}.json"
        try:
            IOLAuth(
                user,
                password,
                tokens_file=tokens_path,
                allow_plain_tokens=settings.allow_plain_tokens,
            ).clear_tokens()
        finally:
            from services.cache import get_client_cached
            cache_key = st.session_state.get("cache_key")
            if cache_key:
                get_client_cached.clear(cache_key)
            get_client_cached.clear()
            st.session_state.clear()

    def build_client(self) -> Tuple[Any | None, Exception | None]:
        from services.cache import build_iol_client

        return build_iol_client()


_provider: AuthenticationProvider = IOLAuthenticationProvider()


def register_auth_provider(provider: AuthenticationProvider) -> None:
    """Registra un proveedor de autenticación alternativo."""

    global _provider
    _provider = provider


def get_auth_provider() -> AuthenticationProvider:
    """Obtiene el proveedor de autenticación actualmente registrado."""

    return _provider


def login(user: str, password: str) -> dict:
    """Wrapper para el login utilizando el proveedor registrado."""

    return _provider.login(user, password)


def logout(user: str, password: str = "") -> None:
    """Wrapper para el logout utilizando el proveedor registrado."""

    _provider.logout(user, password)

