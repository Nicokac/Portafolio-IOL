from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest

from application.portfolio_service import calc_rows


def _make_positions(**overrides: object) -> pd.DataFrame:
    base_row: dict[str, object] = {
        "simbolo": "TEST",
        "mercado": "bcba",
        "cantidad": 2.0,
        "costo_unitario": 100.0,
        "tipo": "Accion",
        "tipo_iol": "Accion",
        "tipo_estandar": "Accion",
        "moneda": "ARS",
        "valorizado": np.nan,
        "ultimoPrecio": np.nan,
        "variacionDiaria": np.nan,
    }
    base_row.update(overrides)
    return pd.DataFrame([base_row])


def test_calc_rows_prioritizes_valorizado_over_other_sources(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    df_pos = _make_positions(valorizado=300.0, ultimoPrecio=140.0)

    def _quote_fn(_mercado: str, _simbolo: str) -> dict[str, object]:
        return {"last": 180.0, "provider": "iol", "moneda_origen": "ARS"}

    result = calc_rows(_quote_fn, df_pos, exclude_syms=[])

    assert math.isclose(result.loc[0, "valor_actual"], 300.0)
    assert math.isclose(result.loc[0, "ultimo"], 150.0)
    assert any("proveedor_utilizado" in rec.message for rec in caplog.records)


def test_calc_rows_uses_ultimo_precio_for_ars_when_valorizado_missing() -> None:
    df_pos = _make_positions(moneda="ARS", ultimoPrecio=140.0)

    def _quote_fn(_mercado: str, _simbolo: str) -> dict[str, object]:
        return {"last": 200.0, "provider": "iol", "moneda_origen": "ARS"}

    result = calc_rows(_quote_fn, df_pos, exclude_syms=[])

    assert math.isclose(result.loc[0, "valor_actual"], 280.0)
    assert math.isclose(result.loc[0, "ultimo"], 140.0)


def test_calc_rows_falls_back_to_cotizacion_quote() -> None:
    df_pos = _make_positions(moneda="USD")

    def _quote_fn(_mercado: str, _simbolo: str) -> dict[str, object]:
        return {"last": 210.0, "provider": "iol", "moneda_origen": "ARS"}

    result = calc_rows(_quote_fn, df_pos, exclude_syms=[])

    assert math.isclose(result.loc[0, "valor_actual"], 420.0)
    assert math.isclose(result.loc[0, "ultimo"], 210.0)


def test_calc_rows_uses_external_last_when_fx_available() -> None:
    df_pos = _make_positions(moneda="USD")

    def _quote_fn(_mercado: str, _simbolo: str) -> dict[str, object]:
        return {
            "last": 5.0,
            "provider": "polygon",
            "moneda_origen": "USD",
            "fx_aplicado": 350.0,
        }

    result = calc_rows(_quote_fn, df_pos, exclude_syms=[])

    assert math.isclose(result.loc[0, "valor_actual"], 10.0)
    assert math.isclose(result.loc[0, "ultimo"], 5.0)


def test_calc_rows_ignores_external_last_without_fx() -> None:
    df_pos = _make_positions(moneda="USD")

    def _quote_fn(_mercado: str, _simbolo: str) -> dict[str, object]:
        return {
            "last": 7.0,
            "provider": "polygon",
            "moneda_origen": "USD",
            "fx_aplicado": None,
        }

    result = calc_rows(_quote_fn, df_pos, exclude_syms=[])

    assert np.isnan(result.loc[0, "valor_actual"])
    assert np.isnan(result.loc[0, "ultimo"])
