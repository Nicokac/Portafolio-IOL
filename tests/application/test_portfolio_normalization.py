from __future__ import annotations

import math

from application.portfolio_service import calc_rows, normalize_positions


IOL_ASSET_TYPES: list[str] = ["Cedear", "Acciones", "Bono", "Letra", "FCI"]


def _get_payload(**overrides):
    base = {
        "simbolo": "GGAL",
        "mercado": "bcba",
        "cantidad": 10,
        "costoUnitario": 100,
        "moneda": "ARS",
        "plazo": "T2",
        "ultimoPrecio": 123.45,
        "variacionDiaria": 1.23,
        "tienePanel": True,
        "riesgo": "Baja",
        "valorizado": 1234.5,
        "titulo": {"tipo": "Acciones", "descripcion": "Acciones Locales"},
    }
    base.update(overrides)
    return {"activos": [base]}


def test_normalize_positions_preserves_metadata():
    df = normalize_positions(_get_payload())

    expected_columns = {
        "simbolo",
        "mercado",
        "cantidad",
        "costo_unitario",
        "moneda",
        "plazo",
        "ultimoPrecio",
        "variacionDiaria",
        "tienePanel",
        "riesgo",
        "titulo_tipo_original",
        "titulo_descripcion_original",
        "tipo",
        "tipo_iol",
        "tipo_estandar",
        "valorizado",
    }

    assert expected_columns.issubset(df.columns)
    row = df.iloc[0]
    assert row["moneda"] == "ARS"
    assert row["plazo"] == "T2"
    assert math.isclose(row["ultimoPrecio"], 123.45)
    assert math.isclose(row["variacionDiaria"], 1.23)
    assert bool(row["tienePanel"]) is True
    assert row["riesgo"] == "Baja"
    assert row["titulo_tipo_original"] == "Acciones"
    assert row["titulo_descripcion_original"] == "Acciones Locales"
    assert row["tipo"] == "Acciones"
    assert row["tipo_iol"] == "Acciones"
    assert row["tipo_estandar"] == "Acciones"
    assert math.isclose(row["valorizado"], 1234.5)


def test_calc_rows_uses_payload_price_and_variation_when_quotes_absent():
    df_pos = normalize_positions(_get_payload())

    def _quote_fn(_market: str, _symbol: str):
        return {}

    df_result = calc_rows(_quote_fn, df_pos, exclude_syms=[])

    assert math.isclose(df_result.loc[0, "ultimo"], 123.45)
    assert math.isclose(df_result.loc[0, "pld_%"], 1.23)

    # valor_actual usa el último disponible
    expected_valor_actual = 10 * 123.45
    assert math.isclose(df_result.loc[0, "valor_actual"], expected_valor_actual)

    assert "tipo_iol" in df_result.columns
    assert df_result.loc[0, "tipo_iol"] == "Acciones"
    assert df_result.loc[0, "tipo_estandar"] == "Acciones"


def test_normalize_positions_preserves_all_iol_types():
    payload = {"activos": []}
    for idx, tipo in enumerate(IOL_ASSET_TYPES):
        override = _get_payload(
            simbolo=f"SYM{idx}",
            titulo={"tipo": tipo, "descripcion": f"{tipo} directo"},
            cantidad=idx + 1,
            costoUnitario=100 + idx,
            ultimoPrecio=120 + idx,
            variacionDiaria=0.5,
            valorizado=float((idx + 1) * (120 + idx)),
        )
        payload["activos"].extend(override["activos"])

    df = normalize_positions(payload)

    assert set(df["tipo"]) == set(IOL_ASSET_TYPES)
    assert set(df["tipo_iol"]) == set(IOL_ASSET_TYPES)
    assert set(df["tipo_estandar"]) == set(IOL_ASSET_TYPES)


def test_calc_rows_preserves_iol_types_for_all_positions():
    payload = {"activos": []}
    for idx, tipo in enumerate(IOL_ASSET_TYPES):
        override = _get_payload(
            simbolo=f"AS{idx}",
            titulo={"tipo": tipo, "descripcion": f"{tipo} directo"},
            cantidad=idx + 2,
            costoUnitario=90 + idx,
            ultimoPrecio=130 + idx,
            variacionDiaria=1.5,
            valorizado=float((idx + 2) * (130 + idx)),
        )
        payload["activos"].extend(override["activos"])

    df_pos = normalize_positions(payload)

    def _quote_fn(_market: str, _symbol: str):
        return {}

    df_result = calc_rows(_quote_fn, df_pos, exclude_syms=[])

    assert set(df_result["tipo"]) == set(IOL_ASSET_TYPES)
    assert set(df_result["tipo_iol"]) == set(IOL_ASSET_TYPES)
    assert set(df_result["tipo_estandar"]) == set(IOL_ASSET_TYPES)
