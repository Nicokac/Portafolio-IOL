from __future__ import annotations

import re
from typing import Optional

# Mapeo explícito de alias comunes de la API de IOL → tipos internos estándar
_ALIASES = {
    r"acci[oó]n(?:es)?": "Acción",
    r"cedear": "CEDEAR",
    r"etf internacional": "ETF",
    r"\betf\b": "ETF",
    r"obligaci[oó]n negociable": "Bono / ON",
    r"bono corporativo": "Bono / ON",
    r"bonos? soberanos": "Bono / ON",
    r"\bbono\b": "Bono / ON",
    r"\bletra[s]?\b": "Bono / ON",
    r"fondo money market": "FCI / Money Market",
    r"money market": "FCI / Money Market",
    r"fondo com[úu]n de inversi[oó]n": "FCI / Money Market",
    r"\bfci\b": "FCI / Money Market",
    r"plazo fijo": "Plazo Fijo",
    r"cauci[óo]n": "Caución",
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
