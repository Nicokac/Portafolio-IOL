from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from shared.asset_type_aliases import normalize_asset_type
from shared.config import settings

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "assets_catalog.json"
_LEGACY_PATH = Path(__file__).with_name("assets_catalog.json")


def _default_catalog_path() -> str:
    if _DATA_PATH.exists():
        return str(_DATA_PATH)
    return str(_LEGACY_PATH)


def _coerce_entry(symbol: str, payload: dict[str, Any]) -> dict[str, str]:
    entry: dict[str, str] = {"simbolo": symbol}
    raw_tipo = payload.get("tipo")
    if raw_tipo is not None:
        entry["tipo"] = str(raw_tipo)
    raw_desc = payload.get("descripcion")
    if raw_desc is not None:
        entry["descripcion"] = str(raw_desc)
    raw_standard = payload.get("tipo_estandar")
    if raw_standard:
        entry["tipo_estandar"] = str(raw_standard)
    else:
        standard = normalize_asset_type(entry.get("tipo") or entry.get("descripcion"))
        if standard:
            entry["tipo_estandar"] = standard
    return entry


def _parse_catalog(data: Any) -> Dict[str, dict[str, str]]:
    catalog: Dict[str, dict[str, str]] = {}
    if isinstance(data, dict):
        for symbol, tipo in data.items():
            sym = str(symbol).strip().upper()
            if not sym:
                continue
            payload: dict[str, Any] = {}
            if tipo is not None:
                payload["tipo"] = tipo
            catalog[sym] = _coerce_entry(sym, payload)
        return catalog
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("simbolo") or item.get("symbol") or item.get("ticker") or "")
            sym = symbol.strip().upper()
            if not sym:
                continue
            payload = dict(item)
            payload["simbolo"] = sym
            catalog[sym] = _coerce_entry(sym, payload)
        return catalog
    logger.warning("Formato inesperado para catálogo de activos: %s", type(data).__name__)
    return {}


@lru_cache(maxsize=1)
def get_asset_catalog(path: str | None = None) -> Dict[str, dict[str, str]]:
    """Carga y normaliza el catálogo centralizado de clasificación de activos."""

    catalog_path = path or settings.secret_or_env("ASSET_CATALOG_PATH", _default_catalog_path())
    try:
        with open(catalog_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("No se pudo leer %s: %s", catalog_path, exc)
        return {}
    return _parse_catalog(data)
