# infrastructure/iol/auth.py
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import requests
import streamlit as st
from cryptography.fernet import Fernet, InvalidToken

from services.diagnostics import run_startup_diagnostics
from services.health import record_diagnostics_snapshot
from shared.config import settings
from shared.errors import InvalidCredentialsError, NetworkError, TimeoutError

logger = logging.getLogger(__name__)
_SESSION_USER_ID_KEY = "iol_current_user_id"
_LAST_USER_STATE_KEY = "last_user_id"

TOKEN_URL = "https://api.invertironline.com/token"
REQ_TIMEOUT = 30
TOKEN_REFRESH_MARGIN = 90  # segundos

FERNET: Fernet | None = None
_iol_key = (settings.tokens_key or "").strip() if isinstance(settings.tokens_key, str) else ""
_fastapi_key = getattr(settings, "fastapi_tokens_key", None)
_fastapi_key = (_fastapi_key or "").strip()

if _fastapi_key and _iol_key and _fastapi_key == _iol_key:
    raise RuntimeError("FASTAPI_TOKENS_KEY must be different from IOL_TOKENS_KEY.")

if _iol_key:
    try:
        FERNET = Fernet(_iol_key.encode())
    except Exception as e:
        logger.warning("Clave de cifrado inválida: %s", e)


def _normalize_user_id(value: Any) -> str | None:
    """Return a normalized string representation for ``value`` if possible."""

    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, bool):  # pragma: no cover - bools are not expected
        return str(value)
    if isinstance(value, (int, float)):
        try:
            if isinstance(value, float) and math.isnan(value):
                return None
        except (TypeError, ValueError):  # pragma: no cover - defensive branch
            return None
        return str(value)
    return None


def _resolve_user_id(tokens: Mapping[str, Any] | Dict[str, Any], fallback: str | None) -> str | None:
    """Extract the most relevant user identifier from ``tokens``."""

    candidates = (
        tokens.get("user_id") if isinstance(tokens, Mapping) else None,
        tokens.get("userId") if isinstance(tokens, Mapping) else None,
        tokens.get("UserId") if isinstance(tokens, Mapping) else None,
        tokens.get("user") if isinstance(tokens, Mapping) else None,
        tokens.get("username") if isinstance(tokens, Mapping) else None,
        tokens.get("userName") if isinstance(tokens, Mapping) else None,
        tokens.get("sub") if isinstance(tokens, Mapping) else None,
    )
    for candidate in candidates:
        normalized = _normalize_user_id(candidate)
        if normalized:
            return normalized
    return _normalize_user_id(fallback)


def _update_session_user_identity(user_id: str | None) -> None:
    """Persist the current user identity in ``st.session_state``."""

    state = getattr(st, "session_state", None)
    if state is None:
        return
    try:
        if user_id is None:
            state.pop(_SESSION_USER_ID_KEY, None)
            state.pop(_LAST_USER_STATE_KEY, None)
        else:
            state[_SESSION_USER_ID_KEY] = user_id
            state[_LAST_USER_STATE_KEY] = user_id
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo actualizar el estado de usuario", exc_info=True)


def get_current_user_id() -> str | None:
    """Return the active user identifier stored in Streamlit's session state."""

    state = getattr(st, "session_state", None)
    if state is None:
        return None
    try:
        user_id = _normalize_user_id(state.get(_SESSION_USER_ID_KEY))
        if user_id:
            return user_id
        return _normalize_user_id(state.get(_LAST_USER_STATE_KEY))
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo obtener el usuario actual", exc_info=True)
        return None


@dataclass
class IOLAuth:
    user: str
    password: str
    tokens_file: Path | str | None = None
    allow_plain_tokens: bool = False

    def __post_init__(self):
        # Usa la ruta de settings por defecto si no se pasa una explícita
        path = Path(self.tokens_file or settings.tokens_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.parent.chmod(0o700)
        except OSError as e:
            logger.exception("No se pudo crear directorio de tokens: %s", e)
        object.__setattr__(self, "tokens_file", path)
        app_env = getattr(settings, "app_env", "dev").lower()
        if self.allow_plain_tokens:
            logger.warning("Plain token storage enabled (development only)")
            if app_env == "prod":
                raise RuntimeError("Plain token storage cannot be enabled in production.")
        if FERNET is None:
            msg = "IOL_TOKENS_KEY no está configurada; los tokens se guardarían sin cifrar."
            if self.allow_plain_tokens:
                logger.warning(msg)
            else:
                raise RuntimeError(msg)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.USER_AGENT})
        self._lock = threading.RLock()
        self._bootstrap_refresh_required = False
        self.tokens: Dict[str, Any] = self._load_tokens() or {}
        if self._needs_bootstrap_refresh(self.tokens):
            self._bootstrap_tokens()

    @property
    def tokens_path(self) -> str:
        return str(self.tokens_file)

    def _save_tokens(self, data: Dict[str, Any]) -> None:
        """Persist token information to disk atomically."""
        now = int(time.time())
        data["timestamp"] = now
        expires_in = data.get("expires_in")
        try:
            if isinstance(expires_in, (int, float)):
                data["expires_at"] = now + int(math.floor(float(expires_in)))
            elif isinstance(expires_in, str) and expires_in.isdigit():
                data["expires_at"] = now + int(expires_in)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            data.pop("expires_at", None)
        try:
            tmp = self.tokens_file.with_suffix(self.tokens_file.suffix + ".tmp")
            content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            if FERNET:
                content = FERNET.encrypt(content)
            tmp.write_bytes(content)
            tmp.replace(self.tokens_file)
            os.chmod(self.tokens_file, 0o600)
        except (OSError, TypeError) as e:
            raise RuntimeError(f"No se pudo guardar el archivo de tokens: {e}") from e

    def _load_tokens(self) -> Dict[str, Any]:
        """Carga tokens desde disco si existen y valida su antigüedad."""
        if self.tokens_file.exists():
            try:
                raw = self.tokens_file.read_bytes()
                if FERNET:
                    raw = FERNET.decrypt(raw)
                data = json.loads(raw.decode("utf-8"))
            except (InvalidToken, json.JSONDecodeError, OSError) as e:
                logger.warning("No se pudieron leer tokens: %s", e)
            else:
                ts = data.get("timestamp")
                ttl_days = getattr(settings, "tokens_ttl_days", 30)
                if ts is None:
                    self._bootstrap_refresh_required = True
                elif ttl_days > 0 and time.time() - ts > ttl_days * 86400:
                    logger.info(
                        "Tokens vencidos; se intentará refresh()",
                        extra={"tokens_file": self.tokens_path},
                    )
                    self._bootstrap_refresh_required = True
                return data
        return {}

    def _token_timestamp_expired(self, data: Dict[str, Any]) -> bool:
        ts = data.get("timestamp")
        if ts is None:
            return True
        try:
            ts_value = float(ts)
        except (TypeError, ValueError):
            return True
        ttl_days = getattr(settings, "tokens_ttl_days", 30)
        if ttl_days <= 0:
            return False
        return (time.time() - ts_value) > ttl_days * 86400

    def _access_token_expired(self, data: Dict[str, Any]) -> bool:
        expires_at = data.get("expires_at")
        if expires_at is None:
            expires_in = data.get("expires_in")
            try:
                if isinstance(expires_in, str) and expires_in.isdigit():
                    expires_at = int(expires_in) + int(data.get("timestamp", 0))
                elif isinstance(expires_in, (int, float)):
                    expires_at = int(data.get("timestamp", 0)) + int(expires_in)
            except (TypeError, ValueError):
                expires_at = None
        try:
            expires_at_value = float(expires_at) if expires_at is not None else None
        except (TypeError, ValueError):
            expires_at_value = None
        if expires_at_value is None:
            return False
        return (time.time() + TOKEN_REFRESH_MARGIN) >= expires_at_value

    def _needs_bootstrap_refresh(self, data: Dict[str, Any]) -> bool:
        if not data:
            return False
        if not data.get("refresh_token"):
            return False
        if self._bootstrap_refresh_required:
            return True
        if not data.get("access_token"):
            return True
        if self._token_timestamp_expired(data):
            return True
        if self._access_token_expired(data):
            return True
        return False

    def _bootstrap_tokens(self) -> None:
        try:
            self.refresh(silent=True)
        except InvalidCredentialsError:
            logger.info(
                "Refresh inválido al inicializar tokens; se requerirá login()",
                extra={"tokens_file": self.tokens_path},
            )
            self.clear_tokens()
        except (TimeoutError, NetworkError) as exc:
            logger.info(
                "Refresh inicial falló (%s); se forzará login()",
                exc,
                extra={"tokens_file": self.tokens_path},
            )
            self.clear_tokens()

    def login(self) -> Dict[str, Any]:
        """Realiza el login contra la API de IOL y guarda los tokens."""
        with self._lock:
            payload = {
                "username": self.user,
                "password": self.password,
                "grant_type": "password",
            }
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            start = time.time()
            try:
                r = self.session.post(TOKEN_URL, data=payload, headers=headers, timeout=REQ_TIMEOUT)
            except (requests.ConnectionError, requests.Timeout) as e:
                logger.warning(
                    "Login IOL falló: %s",
                    e,
                    extra={"tokens_file": self.tokens_path, "result": "error"},
                )
                raise TimeoutError("Fallo de red") from e
            except requests.RequestException as e:
                logger.warning(
                    "Login IOL falló: %s",
                    e,
                    extra={"tokens_file": self.tokens_path, "result": "error"},
                )
                raise NetworkError(str(e)) from e

            if r.status_code in (400, 401):
                logger.warning(
                    f"Auth failed (code={r.status_code})",
                    extra={"tokens_file": self.tokens_path, "result": "error"},
                )
                raise InvalidCredentialsError("Credenciales inválidas")
            try:
                r.raise_for_status()
                self.tokens = r.json() or {}
            except requests.RequestException as e:
                logger.warning(
                    "Login IOL falló: %s",
                    e,
                    extra={"tokens_file": self.tokens_path, "result": "error"},
                )
                raise NetworkError(str(e)) from e

            self._save_tokens(self.tokens)
            _update_session_user_identity(_resolve_user_id(self.tokens, self.user))
            end = time.time()
            logger.info(
                "IOL login ok",
                extra={
                    "tokens_file": self.tokens_path,
                    "result": "ok",
                    "elapsed_ms": int((end - start) * 1000),
                },
            )
            st.session_state["iol_login_ok_ts"] = end
            st.session_state.pop("iol_startup_metric_logged", None)

            analysis_logger = logging.getLogger("analysis")
            try:
                diagnostics = run_startup_diagnostics(tokens=self.tokens)
            except Exception as exc:  # pragma: no cover - defensive logging path
                analysis_logger.info(
                    "startup_diagnostics_failed",
                    extra={
                        "component": "startup_diagnostics",
                        "error": str(exc),
                    },
                )
            else:
                analysis_logger.info(
                    "startup_diagnostics",
                    extra={
                        "component": diagnostics.get("component", "startup_diagnostics"),
                        "status": diagnostics.get("status"),
                        "latency_ms": diagnostics.get("latency"),
                        "details": diagnostics,
                    },
                )
                try:
                    record_diagnostics_snapshot(diagnostics)
                except Exception:  # pragma: no cover - defensive logging path
                    logger.exception("No se pudo registrar el diagnóstico de inicio")
            return self.tokens

    def refresh(self, *, silent: bool = False) -> Dict[str, Any]:
        """Renueva el access_token utilizando el refresh_token."""
        with self._lock:
            if not self.tokens.get("refresh_token"):
                logger.info(
                    "Sin refresh_token; ejecutando login()",
                    extra={"tokens_file": self.tokens_path},
                )
                return self.login()
            payload = {
                "refresh_token": self.tokens["refresh_token"],
                "grant_type": "refresh_token",
            }
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            start = time.time()
            try:
                r = self.session.post(TOKEN_URL, data=payload, headers=headers, timeout=REQ_TIMEOUT)
            except (requests.ConnectionError, requests.Timeout) as e:
                log = logger.debug if silent else logger.warning
                log(
                    "Refresh falló: %s",
                    e,
                    extra={"tokens_file": self.tokens_path, "result": "error"},
                )
                raise TimeoutError("Fallo de red") from e
            except requests.RequestException as e:
                log = logger.debug if silent else logger.warning
                log(
                    "Refresh falló: %s",
                    e,
                    extra={"tokens_file": self.tokens_path, "result": "error"},
                )
                raise NetworkError(str(e)) from e

            if r.status_code in (400, 401):
                log = logger.info if silent else logger.warning
                log(
                    f"Auth failed (code={r.status_code})",
                    extra={"tokens_file": self.tokens_path, "result": "error"},
                )
                self.clear_tokens()
                raise InvalidCredentialsError("Credenciales inválidas")
            try:
                r.raise_for_status()
                self.tokens = r.json() or {}
            except requests.RequestException as e:
                log = logger.debug if silent else logger.warning
                log(
                    "Refresh falló: %s",
                    e,
                    extra={"tokens_file": self.tokens_path, "result": "error"},
                )
                raise NetworkError(str(e)) from e

            self._save_tokens(self.tokens)
            _update_session_user_identity(_resolve_user_id(self.tokens, self.user))
            logger.info(
                "IOL refresh ok",
                extra={
                    "tokens_file": self.tokens_path,
                    "result": "ok",
                    "elapsed_ms": int((time.time() - start) * 1000),
                },
            )
            return self.tokens

    def ensure_token(self, *, silent: bool = True) -> Dict[str, Any]:
        """Garantiza que exista un access token válido, refrescándolo si expira."""
        with self._lock:
            tokens = dict(self.tokens)
            if not tokens.get("access_token"):
                return self.login()
            if self._token_timestamp_expired(tokens) or self._access_token_expired(tokens):
                return self.refresh(silent=silent)
            return tokens

    def clear_tokens(self) -> None:
        """Elimina el archivo de tokens para forzar reautenticación la próxima vez."""
        with self._lock:
            self.tokens = {}
            self._bootstrap_refresh_required = False
            _update_session_user_identity(None)
            try:
                os.remove(self.tokens_file)
                logger.info("Tokens eliminados: %s", self.tokens_file)
            except FileNotFoundError:
                logger.info("Tokens ya no existían: %s", self.tokens_file)
            except OSError as e:
                logger.exception("No se pudo eliminar tokens en %s: %s", self.tokens_file, e)

    def auth_header(self) -> Dict[str, str]:
        """Devuelve el header Authorization actual, refrescando tokens si es necesario."""
        tokens = self.ensure_token(silent=True)
        token = tokens.get("access_token") if isinstance(tokens, Mapping) else None
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}
