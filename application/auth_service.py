from __future__ import annotations

"""Servicios de autenticación para la aplicación."""

from pathlib import Path
from typing import Protocol, Tuple, Any
import re
import hashlib
from uuid import uuid4
import logging

import streamlit as st

from infrastructure.iol.auth import IOLAuth, InvalidCredentialsError, NetworkError
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
        user_hash = hashlib.sha256(user.encode()).hexdigest()[:8]
        tokens_path = Path("tokens") / f"{sanitized}-{user_hash}.json"
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
        user_hash = hashlib.sha256(user.encode()).hexdigest()[:8]
        tokens_path = Path("tokens") / f"{sanitized}-{user_hash}.json"
        try:
            IOLAuth(
                user,
                password,
                tokens_file=tokens_path,
                allow_plain_tokens=settings.allow_plain_tokens,
            ).clear_tokens()
        finally:
            from services.cache import (
                get_client_cached,
                fetch_portfolio,
                fetch_quotes_bulk,
                fetch_fx_rates,
            )
            cache_key = st.session_state.get("cache_key")
            if cache_key:
                get_client_cached.clear(cache_key)
            get_client_cached.clear()
            theme = st.session_state.get("ui_theme")
            for key in (
                "authenticated",
                "IOL_USERNAME",
                "IOL_PASSWORD",
                "client_salt",
                "session_id",
                "last_refresh",
                "cache_key",
                "fx_rates",
                "fx_rates_ts",
                "fx_rates_error",
                "portfolio_tab",
                "controls_snapshot",
                "quotes_hist",
            ):
                st.session_state.pop(key, None)
            cache.pop("tokens_file", None)
            cache.clear()
            fetch_portfolio.clear()
            fetch_quotes_bulk.clear()
            fetch_fx_rates.clear()
            if theme is not None:
                st.session_state["ui_theme"] = theme

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
    tokens = _provider.login(user, password)
    if "client_salt" not in st.session_state:
        st.session_state["client_salt"] = uuid4().hex
    return tokens


def logout(user: str, password: str = "") -> None:
    """Wrapper para el logout utilizando el proveedor registrado."""

    try:
        _provider.logout(user, password)
    except Exception as e:  # pragma: no cover - defensive
        logging.getLogger(__name__).warning("Error al limpiar tokens: %s", e)
        st.session_state["logout_error"] = str(e)
    else:
        st.session_state["logout_done"] = True
    finally:
        st.session_state["force_login"] = True
        st.rerun()

