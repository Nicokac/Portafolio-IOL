from shared.asset_type_aliases import normalize_asset_type


def test_alias_normalization_basic() -> None:
    assert normalize_asset_type("Obligación Negociable") == "Bono / ON"
    assert normalize_asset_type("Fondo Money Market") == "FCI / Money Market"
    assert normalize_asset_type("ETF Internacional") == "ETF"
    assert normalize_asset_type("Acción") == "Acción"
    assert normalize_asset_type("Plazo Fijo") == "Plazo Fijo"


def test_alias_normalization_unknown_or_empty() -> None:
    assert normalize_asset_type("") is None
    assert normalize_asset_type(None) is None
    assert normalize_asset_type("Instrumento Desconocido") is None
