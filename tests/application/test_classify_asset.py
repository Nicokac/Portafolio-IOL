from application.portfolio_service import classify_asset


def test_classify_asset_returns_tipo_from_payload() -> None:
    item = {
        "simbolo": "GGAL",
        "titulo": {"tipo": "Acciones", "descripcion": "Acciones ordinarias"},
    }

    result = classify_asset(item)

    assert result["tipo"] == "Acciones"
    assert result["tipo_estandar"] == "Acciones"
    assert result["tipo_iol"] == "Acciones"


def test_classify_asset_defaults_to_nd_when_tipo_missing() -> None:
    item = {
        "simbolo": "BND",
        "titulo": {"descripcion": "Obligacion Negociable"},
    }

    result = classify_asset(item)

    assert result["tipo"] == "N/D"
    assert result["tipo_estandar"] == "N/D"
    assert result["tipo_iol"] == "N/D"
