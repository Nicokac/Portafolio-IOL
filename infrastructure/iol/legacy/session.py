from __future__ import annotations

import logging
import os
import threading
import time
import random
from typing import Optional

import streamlit as st
import requests
from iolConn import Iol
from iolConn.common.exceptions import NoAuthException

from shared.time_provider import TimeProvider
from shared.errors import InvalidCredentialsError

logger = logging.getLogger(__name__)


AUTH_FAILURE_COOLDOWN_SECONDS = float(
    os.getenv("LEGACY_AUTH_FAILURE_COOLDOWN_SECONDS", "0") or 0
)


class LegacySession:
    """Singleton that provides a shared authenticated legacy ``Iol`` session."""

    _instance: "LegacySession" | None = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._iol: Optional[Iol] = None
        self._ready = False
        self._user: str | None = None
        self._password: str | None = None
        self._legacy_auth_unavailable = False
        self._legacy_auth_unavailable_at: float | None = None
        self._tokens_snapshot: tuple[str | None, str | None] | None = None

    # ------------------------------------------------------------------
    # Singleton helpers
    # ------------------------------------------------------------------
    @classmethod
    def get(cls) -> "LegacySession":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_auth_unavailable(self) -> bool:
        with self._lock:
            return self._legacy_auth_unavailable

    def invalidate(self) -> None:
        with self._lock:
            self._iol = None
            self._ready = False

    def ensure_authenticated(self, user: str, password: str, auth) -> Optional[Iol]:
        """Return a shared authenticated ``Iol`` instance or ``None`` if auth fails."""

        norm_user = (user or "").strip()
        norm_password = (password or "").strip()
        tokens = getattr(auth, "tokens", {}) if auth is not None else {}
        bearer = tokens.get("access_token") if isinstance(tokens, dict) else None
        refresh = tokens.get("refresh_token") if isinstance(tokens, dict) else None
        tokens_snapshot = (str(bearer) if bearer else None, str(refresh) if refresh else None)

        with self._lock:
            creds_changed = (self._user, self._password) != (norm_user, norm_password)
            tokens_changed = self._tokens_snapshot != tokens_snapshot

            if self._legacy_auth_unavailable:
                can_retry = creds_changed or tokens_changed
                if not can_retry and self._legacy_auth_unavailable_at is not None:
                    cooldown = AUTH_FAILURE_COOLDOWN_SECONDS
                    if cooldown > 0:
                        elapsed = time.monotonic() - self._legacy_auth_unavailable_at
                        can_retry = elapsed >= cooldown

                if not can_retry:
                    st.session_state["legacy_auth_unavailable"] = True
                    return None

                self._legacy_auth_unavailable = False
                self._legacy_auth_unavailable_at = None
                st.session_state.pop("legacy_auth_unavailable", None)

            if creds_changed or tokens_changed:
                self._ready = False
                self._iol = None
                self._user = norm_user
                self._password = norm_password
                self._tokens_snapshot = tokens_snapshot

            if self._ready and self._iol is not None:
                return self._iol

            try:
                session = self._build_session(norm_user, norm_password, bearer, refresh)
                self._iol = session
                self._ready = True
                self._user = norm_user
                self._password = norm_password
                self._tokens_snapshot = tokens_snapshot
                self._legacy_auth_unavailable = False
                self._legacy_auth_unavailable_at = None
                st.session_state.pop("legacy_auth_unavailable", None)
                return self._iol
            except InvalidCredentialsError:
                self._mark_auth_unavailable()
                return None
            except NoAuthException as exc:
                logger.warning("Legacy IOL auth failed: %s", exc)
                self._mark_auth_unavailable()
                return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _mark_auth_unavailable(self) -> None:
        self._ready = False
        self._iol = None
        self._legacy_auth_unavailable = True
        self._legacy_auth_unavailable_at = time.monotonic()
        st.session_state["legacy_auth_unavailable"] = True

    def _build_session(
        self,
        user: str,
        password: str,
        bearer: str | None,
        refresh: str | None,
    ) -> Iol:
        session = Iol(user, password)
        if bearer and refresh:
            session.bearer = bearer
            session.refresh_token = refresh
            bearer_time = TimeProvider.now_datetime().replace(tzinfo=None)
            session.bearer_time = bearer_time
        elif not password:
            st.session_state["force_login"] = True
            raise InvalidCredentialsError("Token inválido")

        try:
            session.gestionar()
            return session
        except NoAuthException:
            if not password:
                st.session_state["force_login"] = True
                raise InvalidCredentialsError("Token inválido")

        password_session = Iol(user, password)
        try:
            password_session.gestionar()
        except NoAuthException:
            st.session_state["force_login"] = True
            raise InvalidCredentialsError("Credenciales inválidas")

        return password_session

    def fetch_with_backoff(
        self,
        market: str,
        symbol: str,
        *,
        panel: str | None = None,
        auth_user: str,
        auth_password: str,
        auth,
    ) -> tuple[Optional[dict], bool]:
        """Fetch quote data handling retries for HTTP 429."""

        session = self.ensure_authenticated(auth_user, auth_password, auth)
        if session is None:
            st.session_state["legacy_auth_unavailable"] = True
            return None, True

        delays = (0.5, 1.0, 2.0)
        last_exc: Exception | None = None
        for attempt, base_delay in enumerate(delays):
            try:
                data = session.price_to_json(mercado=market, simbolo=symbol, panel=panel)
                return data if isinstance(data, dict) else None, False
            except NoAuthException:
                self.invalidate()
                session = self.ensure_authenticated(auth_user, auth_password, auth)
                if session is None:
                    return None, True
            except requests.HTTPError as exc:
                last_exc = exc
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code == 429 and attempt < len(delays) - 1:
                    wait = base_delay + random.uniform(0, base_delay)
                    time.sleep(wait)
                    continue
                raise
            except requests.RequestException as exc:
                last_exc = exc
                raise

        if last_exc:
            raise last_exc
        return None, False


__all__ = ["LegacySession"]
