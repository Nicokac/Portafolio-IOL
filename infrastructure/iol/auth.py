# infrastructure/iol/auth.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import logging
import json
import time
from typing import Dict, Any

import requests
from cryptography.fernet import Fernet

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

    def login(self) -> Dict[str, Any]:
        """Realiza el login contra la API de IOL y guarda los tokens."""
        payload = {"username": self.user, "password": self.password, "grant_type": "password"}
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": settings.USER_AGENT,
        }
        start = time.time()
        try:
            r = requests.post(TOKEN_URL, data=payload, headers=headers, timeout=REQ_TIMEOUT)
            r.raise_for_status()
            tokens: Dict[str, Any] = r.json() or {}
        except requests.RequestException as e:
            logger.warning("Login IOL falló: %s", e)
            return {}
        self._save_tokens(tokens)
        logger.info("IOL login ok en %d ms", int((time.time() - start) * 1000))
        return tokens

    def clear_tokens(self) -> None:
        """Elimina el archivo de tokens para forzar reautenticación la próxima vez."""
        try:
            os.remove(self.tokens_file)
            logger.info("Tokens eliminados: %s", self.tokens_file)
        except FileNotFoundError:
            logger.info("Tokens ya no existían: %s", self.tokens_file)
        except OSError as e:
            logger.exception("No se pudo eliminar tokens en %s: %s", self.tokens_file, e)
