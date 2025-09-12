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
        try:
            tmp = self.tokens_file.with_suffix(self.tokens_file.suffix + ".tmp")
            content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            if FERNET:
                content = FERNET.encrypt(content)
            tmp.write_bytes(content)
            tmp.replace(self.tokens_file)
            os.chmod(self.tokens_file, 0o600)
        except (OSError, TypeError) as e:
            logger.exception("No se pudo guardar el archivo de tokens: %s", e)

    def _load_tokens(self) -> Dict[str, Any]:
        """Carga tokens desde disco si existen."""
        if self.tokens_file.exists():
            try:
                raw = self.tokens_file.read_bytes()
                if FERNET:
                    raw = FERNET.decrypt(raw)
                return json.loads(raw.decode("utf-8"))
            except (InvalidToken, json.JSONDecodeError, OSError) as e:
                logger.warning("No se pudieron leer tokens: %s", e)
        return {}

    def login(self) -> Dict[str, Any]:
        """Realiza el login contra la API de IOL y guarda los tokens."""
        with self._lock:
            payload = {"username": self.user, "password": self.password, "grant_type": "password"}
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            start = time.time()
            try:
                r = self.session.post(TOKEN_URL, data=payload, headers=headers, timeout=REQ_TIMEOUT)
                r.raise_for_status()
                self.tokens = r.json() or {}
            except requests.RequestException as e:
                logger.warning("Login IOL falló: %s", e)
                return {}
            self._save_tokens(self.tokens)
            logger.info("IOL login ok en %d ms", int((time.time() - start) * 1000))
            return self.tokens

    def refresh(self) -> Dict[str, Any]:
        """Renueva el access_token utilizando el refresh_token."""
        with self._lock:
            if not self.tokens.get("refresh_token"):
                logger.info("Sin refresh_token; ejecutando login()")
                return self.login()
            payload = {"refresh_token": self.tokens["refresh_token"], "grant_type": "refresh_token"}
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            start = time.time()
            try:
                r = self.session.post(TOKEN_URL, data=payload, headers=headers, timeout=REQ_TIMEOUT)
                r.raise_for_status()
                self.tokens = r.json() or {}
            except requests.RequestException as e:
                logger.warning("Refresh falló; intentando login(): %s", e)
                return self.login()
            self._save_tokens(self.tokens)
            logger.info("IOL refresh ok en %d ms", int((time.time() - start) * 1000))
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
