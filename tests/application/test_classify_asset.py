from application.portfolio_service import classify_asset


def test_classify_asset_uses_description_for_bond_aliases() -> None:
    item = {
        "simbolo": "XZ123",
        "titulo": {"tipo": "Otros", "descripcion": "Obligacion Negociable Clase A"},
    }

    result = classify_asset(item)

    assert result["tipo_estandar"] == "Bono / ON"
    assert result["tipo"] == "Bono / ON"
    assert result["tipo_iol"] == "Obligacion Negociable Clase A"


def test_classify_asset_detects_fci_from_description() -> None:
    item = {
        "simbolo": "MM123",
        "titulo": {"tipo": "Otros", "descripcion": "Fondo Money Market T+1"},
    }

    result = classify_asset(item)

    assert result["tipo_estandar"] == "FCI / Money Market"
    assert result["tipo"] == "FCI / Money Market"
    assert result["tipo_iol"] == "Fondo Money Market T+1"


def test_classify_asset_preserves_unknown_label_and_falls_back_to_symbol() -> None:
    item = {
        "simbolo": "AL30",
        "titulo": {"tipo": "Renta Especial", "descripcion": "Raro"},
    }

    result = classify_asset(item)

    assert result["tipo_iol"] == "Renta Especial"
    assert result["tipo_estandar"] == "Bono / ON"
