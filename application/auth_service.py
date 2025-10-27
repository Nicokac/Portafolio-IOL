from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Protocol, Tuple
from uuid import uuid4

import streamlit as st

from infrastructure.iol.auth import IOLAuth
from services.auth import revoke_token
from services.performance_timer import performance_timer
from shared.cache import cache
from shared.config import settings
from shared.fragment_state import persist_fragment_state_snapshot
from shared.debug.rerun_trace import mark_event, safe_rerun

"""Servicios de autenticación para la aplicación."""


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
        self._user = user
        sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", user)
        user_hash = hashlib.sha256(user.encode()).hexdigest()[:12]
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

    def logout(self, user: str = "", password: str = "") -> None:
        user = user or getattr(self, "_user", "")
        tokens_file = cache.get("tokens_file")
        if user:
            sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", user)
            user_hash = hashlib.sha256(user.encode()).hexdigest()[:12]
            tokens_path = Path("tokens") / f"{sanitized}-{user_hash}.json"
        elif tokens_file:
            tokens_path = Path(tokens_file)
        else:
            tokens_path = None
        try:
            if tokens_path:
                IOLAuth(
                    user or "",
                    password,
                    tokens_file=tokens_path,
                    allow_plain_tokens=settings.allow_plain_tokens,
                ).clear_tokens()
        finally:
            from services.cache import (
                fetch_fx_rates,
                fetch_portfolio,
                fetch_quotes_bulk,
                get_client_cached,
            )

            cache_key = st.session_state.get("cache_key")
            if cache_key:
                cache_user = user or st.session_state.get("IOL_USERNAME", "")
                tokens_arg = tokens_path if tokens_path is not None else cache.get("tokens_file")
                get_client_cached.clear(cache_key, cache_user, tokens_arg)
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
                "auth_token",
                "auth_token_claims",
                "auth_token_refreshed_at",
            ):
                st.session_state.pop(key, None)
            cache.pop("tokens_file", None)
            cache.clear()
            fetch_portfolio.clear()
            fetch_quotes_bulk.clear()
            fetch_fx_rates.clear()
            if theme is not None:
                st.session_state["ui_theme"] = theme
            self._user = None

    def build_client(self) -> Tuple[Any | None, Exception | None]:
        from services.cache import build_iol_client

        return build_iol_client(getattr(self, "_user", None))


_provider: AuthenticationProvider = IOLAuthenticationProvider()


def _mask_username(user: str) -> str:
    sanitized = (user or "").strip()
    if not sanitized:
        return "anon"
    if len(sanitized) <= 3:
        return sanitized[:1] + "**"
    return f"{sanitized[:3]}***"


def register_auth_provider(provider: AuthenticationProvider) -> None:
    """Registra un proveedor de autenticación alternativo."""

    global _provider
    _provider = provider


def get_auth_provider() -> AuthenticationProvider:
    """Obtiene el proveedor de autenticación actualmente registrado."""

    return _provider


def login(user: str, password: str) -> dict:
    """Wrapper para el login utilizando el proveedor registrado."""
    telemetry: dict[str, object] = {
        "user": _mask_username(user),
        "status": "success",
    }
    with performance_timer("login_iol", extra=telemetry):
        try:
            tokens = _provider.login(user, password)
        except Exception:
            telemetry["status"] = "error"
            raise
    if "client_salt" not in st.session_state:
        st.session_state["client_salt"] = uuid4().hex
    return tokens


def logout(user: str = "", password: str = "") -> None:
    """Wrapper para el logout utilizando el proveedor registrado."""

    try:
        persist_fragment_state_snapshot()
    except Exception:  # pragma: no cover - defensive safeguard
        logging.getLogger(__name__).debug(
            "No se pudo persistir el estado de fragmentos antes de logout",
            exc_info=True,
        )
    user = user or st.session_state.get("IOL_USERNAME", "")
    revoke_token(st.session_state.get("auth_token"))
    try:
        _provider.logout(user, password)
    except Exception as e:  # pragma: no cover - defensive
        logging.getLogger(__name__).warning("Error al limpiar tokens: %s", e)
        st.session_state["logout_error"] = str(e)
    else:
        st.session_state["logout_done"] = True
    finally:
        st.session_state["force_login"] = True
        mark_event("rerun", "auth_logout_force_login")
        safe_rerun("auth_logout_force_login")
