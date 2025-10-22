import json
from pathlib import Path

from shared.asset_type_aliases import normalize_asset_type


_ALLOWED_TYPES = {
    "Acción",
    "CEDEAR",
    "ETF",
    "Bono / ON",
    "FCI / Money Market",
    "Plazo Fijo",
    "Caución",
    None,
}


def test_catalog_asset_types_are_standardized():
    data = json.loads(Path("data/assets_catalog.json").read_text(encoding="utf-8"))
    for item in data:
        assert isinstance(item, dict), f"Entrada inválida en catálogo: {item!r}"
        tipo = item.get("tipo_estandar")
        assert tipo in _ALLOWED_TYPES, f"Tipo no estandarizado: {tipo}"
        if tipo is not None:
            assert normalize_asset_type(tipo) == tipo
