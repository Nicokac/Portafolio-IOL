from application.portfolio_service import classify_asset


def test_classify_asset_uses_description_for_bond_aliases() -> None:
    item = {
        "simbolo": "XZ123",
        "titulo": {"tipo": "Otros", "descripcion": "Obligacion Negociable Clase A"},
    }

    assert classify_asset(item) == "Bono"


def test_classify_asset_detects_fci_from_description() -> None:
    item = {
        "simbolo": "MM123",
        "titulo": {"tipo": "Otros", "descripcion": "Fondo Money Market T+1"},
    }

    assert classify_asset(item) == "FCI"
