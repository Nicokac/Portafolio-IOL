# infrastructure/iol/auth.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import logging

from shared.config import settings

logger = logging.getLogger(__name__)

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
        except Exception:
            pass
        object.__setattr__(self, "tokens_file", path)

    @property
    def tokens_path(self) -> str:
        return str(self.tokens_file)

    def clear_tokens(self) -> None:
        """Elimina el archivo de tokens para forzar reautenticación la próxima vez."""
        try:
            os.remove(self.tokens_file)
            logger.info("Tokens eliminados: %s", self.tokens_file)
        except FileNotFoundError:
            logger.info("Tokens ya no existían: %s", self.tokens_file)
        except Exception as e:
            logger.warning("No se pudo eliminar tokens en %s: %s", self.tokens_file, e)
