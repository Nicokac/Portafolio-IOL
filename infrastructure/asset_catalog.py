from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Dict

from shared.config import settings

logger = logging.getLogger(__name__)

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "assets_catalog.json")


@lru_cache(maxsize=1)
def get_asset_catalog(path: str | None = None) -> Dict[str, str]:
    """Carga un catálogo centralizado de clasificación de activos.

    El archivo se busca en ``ASSET_CATALOG_PATH`` o en un JSON local
    ``assets_catalog.json`` dentro del módulo de infraestructura.
    Devuelve un ``dict`` con símbolos en mayúsculas mapeados al tipo de
    activo correspondiente. Si el archivo no existe o es inválido, se
    retorna un ``dict`` vacío.
    """
    path = path or settings.secret_or_env("ASSET_CATALOG_PATH", DEFAULT_PATH)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                return {str(k).upper(): str(v) for k, v in data.items()}
    except (OSError, json.JSONDecodeError) as e:
        logger.exception("No se pudo leer %s: %s", path, e)
    return {}
