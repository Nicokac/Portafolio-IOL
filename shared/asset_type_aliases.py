from __future__ import annotations

import re
from typing import Optional

# Mapeo explícito de alias comunes de la API de IOL → tipos internos estándar
_ALIASES = {
    "acci[oó]n": "Acción",
    "acciones?": "Acción",
    "cedear": "CEDEAR",
    "etf": "ETF",
    "etf internacional": "ETF",
    "obligaci[oó]n negociable": "Bono / ON",
    "bono": "Bono",
    "bonos soberanos": "Bono",
    "bono corporativo": "Bono / ON",
    "fondo money market": "FCI / Money Market",
    "fondo com[úu]n de inversi[oó]n": "FCI",
    "plazo fijo": "Plazo Fijo",
    "cauci[óo]n": "Caución",
}


def normalize_asset_type(raw: Optional[str]) -> Optional[str]:
    """Normaliza un tipo de activo o descripción proveniente de IOL a una categoría estándar."""

    if not raw:
        return None

    t = raw.strip().lower()
    for pattern, normalized in _ALIASES.items():
        if re.search(pattern, t):
            return normalized
    return None
