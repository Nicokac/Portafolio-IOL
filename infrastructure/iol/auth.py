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

from shared.config import settings

logger = logging.getLogger(__name__)

TOKEN_URL = "https://api.invertironline.com/token"
REQ_TIMEOUT = 30

@dataclass
class IOLAuth:
    user: str
    password: str
    tokens_file: Path | str | None = None

    def __post_init__(self):
        # Usa la ruta de settings por defecto si no se pasa una explícita
        path = Path(self.tokens_file or settings.tokens_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.exception("No se pudo crear directorio de tokens: %s", e)
        object.__setattr__(self, "tokens_file", path)

    @property
    def tokens_path(self) -> str:
        return str(self.tokens_file)

    def _save_tokens(self, data: Dict[str, Any]) -> None:
        """Persist token information to disk atomically."""
        try:
            tmp = self.tokens_file.with_suffix(self.tokens_file.suffix + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
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
