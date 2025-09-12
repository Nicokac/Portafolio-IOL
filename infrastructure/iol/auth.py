# infrastructure/iol/auth.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import logging
import json
import time
import threading
from typing import Dict, Any

import requests
from cryptography.fernet import Fernet, InvalidToken

from shared.config import settings

logger = logging.getLogger(__name__)

TOKEN_URL = "https://api.invertironline.com/token"
REQ_TIMEOUT = 30

FERNET: Fernet | None = None
if settings.tokens_key:
    try:
        FERNET = Fernet(settings.tokens_key.encode())
    except Exception as e:
        logger.warning("Clave de cifrado inválida: %s", e)


class InvalidCredentialsError(Exception):
    """Se lanza cuando el usuario o contraseña son inválidos."""


class NetworkError(Exception):
    """Se lanza ante problemas de conectividad con la API."""

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
        if FERNET is None:
            msg = "IOL_TOKENS_KEY no está configurada; los tokens se guardarían sin cifrar."
            if self.allow_plain_tokens:
                logger.warning(msg)
            else:
                raise RuntimeError(msg)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.USER_AGENT})
        self._lock = threading.RLock()
        self.tokens: Dict[str, Any] = self._load_tokens() or {}

    @property
    def tokens_path(self) -> str:
        return str(self.tokens_file)

    def _save_tokens(self, data: Dict[str, Any]) -> None:
        """Persist token information to disk atomically."""
        data["timestamp"] = int(time.time())
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
                if ts is None or (ttl_days > 0 and time.time() - ts > ttl_days * 86400):
                    logger.info(
                        "Tokens vencidos; ejecutando login()",
                        extra={"tokens_file": self.tokens_path},
                    )
                    self.clear_tokens()
                    return self.login()
                return data
        return {}

    def login(self) -> Dict[str, Any]:
        """Realiza el login contra la API de IOL y guarda los tokens."""
        with self._lock:
            payload = {"username": self.user, "password": self.password, "grant_type": "password"}
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
                raise NetworkError("Fallo de red") from e
            except requests.RequestException as e:
                logger.warning(
                    "Login IOL falló: %s",
                    e,
                    extra={"tokens_file": self.tokens_path, "result": "error"},
                )
                raise NetworkError(str(e)) from e

            if r.status_code in (400, 401):
                logger.warning(
                    "Credenciales inválidas en login IOL: %s",
                    r.text,
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
                raise NetworkError("Fallo de red") from e

            self._save_tokens(self.tokens)
            logger.info(
                "IOL login ok",
                extra={
                    "tokens_file": self.tokens_path,
                    "result": "ok",
                    "elapsed_ms": int((time.time() - start) * 1000),
                },
            )
            return self.tokens

    def refresh(self) -> Dict[str, Any]:
        """Renueva el access_token utilizando el refresh_token."""
        with self._lock:
            if not self.tokens.get("refresh_token"):
                logger.info(
                    "Sin refresh_token; ejecutando login()",
                    extra={"tokens_file": self.tokens_path},
                )
                return self.login()
            payload = {"refresh_token": self.tokens["refresh_token"], "grant_type": "refresh_token"}
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            start = time.time()
            try:
                r = self.session.post(TOKEN_URL, data=payload, headers=headers, timeout=REQ_TIMEOUT)
            except (requests.ConnectionError, requests.Timeout) as e:
                logger.warning(
                    "Refresh falló: %s",
                    e,
                    extra={"tokens_file": self.tokens_path, "result": "error"},
                )
                raise NetworkError("Fallo de red") from e
            except requests.RequestException as e:
                logger.warning(
                    "Refresh falló: %s",
                    e,
                    extra={"tokens_file": self.tokens_path, "result": "error"},
                )
                raise NetworkError(str(e)) from e

            if r.status_code in (400, 401):
                logger.warning(
                    "Credenciales inválidas en refresh IOL: %s",
                    r.text,
                    extra={"tokens_file": self.tokens_path, "result": "error"},
                )
                raise InvalidCredentialsError("Credenciales inválidas")
            try:
                r.raise_for_status()
                self.tokens = r.json() or {}
            except requests.RequestException as e:
                logger.warning(
                    "Refresh falló: %s",
                    e,
                    extra={"tokens_file": self.tokens_path, "result": "error"},
                )
                raise NetworkError("Fallo de red") from e

            self._save_tokens(self.tokens)
            logger.info(
                "IOL refresh ok",
                extra={
                    "tokens_file": self.tokens_path,
                    "result": "ok",
                    "elapsed_ms": int((time.time() - start) * 1000),
                },
            )
            return self.tokens

    def clear_tokens(self) -> None:
        """Elimina el archivo de tokens para forzar reautenticación la próxima vez."""
        with self._lock:
            self.tokens = {}
            try:
                os.remove(self.tokens_file)
                logger.info("Tokens eliminados: %s", self.tokens_file)
            except FileNotFoundError:
                logger.info("Tokens ya no existían: %s", self.tokens_file)
            except OSError as e:
                logger.exception("No se pudo eliminar tokens en %s: %s", self.tokens_file, e)

    def auth_header(self) -> Dict[str, str]:
        """Devuelve el header Authorization actual, realizando login si es necesario."""
        if not self.tokens.get("access_token"):
            self.login()
        token = self.tokens.get("access_token")
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}
