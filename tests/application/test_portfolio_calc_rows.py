import math

from application.portfolio_service import calc_rows, normalize_positions


def test_calc_rows_uses_valorizado_when_quotes_absent() -> None:
    payload = {
        "activos": [
            {
                "simbolo": "BONO1",
                "mercado": "bcba",
                "cantidad": 2,
                "costoUnitario": 100,
                "valorizado": 260,
                "variacionDiaria": 1.5,
                "titulo": {"tipo": "Accion", "descripcion": "Accion Local"},
            }
        ]
    }

    df_pos = normalize_positions(payload)

    def _quote_fn(_mercado: str, _simbolo: str):
        return {}

    df_result = calc_rows(_quote_fn, df_pos, exclude_syms=[])

    assert math.isclose(df_result.loc[0, "valor_actual"], 260.0)
    assert math.isclose(df_result.loc[0, "pl"], 60.0)
    assert math.isclose(df_result.loc[0, "pl_%"], 30.0)
    assert math.isclose(df_result.loc[0, "pld_%"], 1.5)
    assert df_result.loc[0, "tipo_iol"] == "Accion"
    assert df_result.loc[0, "tipo_estandar"] == "Accion"


def test_calc_rows_preserves_description_when_used_for_classification() -> None:
    payload = {
        "activos": [
            {
                "simbolo": "MM123",
                "mercado": "bcba",
                "cantidad": 1,
                "costoUnitario": 10,
                "ultimoPrecio": 11,
                "variacionDiaria": 0.0,
                "titulo": {"tipo": "Otros", "descripcion": "Fondo Money Market"},
            }
        ]
    }

    df_pos = normalize_positions(payload)

    def _quote_fn(_mercado: str, _simbolo: str):
        return {}

    df_result = calc_rows(_quote_fn, df_pos, exclude_syms=[])

    assert df_result.loc[0, "tipo_iol"] == "Otros"
    assert df_result.loc[0, "tipo_estandar"] == "Otros"
