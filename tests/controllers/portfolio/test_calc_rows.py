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


def test_calc_rows_detects_bond_scale_from_portfolio_payload() -> None:
    bond_rows = [
        {
            "simbolo": "AL30",
            "mercado": "bcba",
            "cantidad": 1200.0,
            "costo_unitario": 6_500.0,
            "tipo": "Bonos Soberanos",
            "tipo_iol": "Bonos Soberanos",
            "tipo_estandar": "Bonos Soberanos",
            "moneda": "ARS",
            "valorizado": 8_100_000.0,
            "ultimoPrecio": 6_750.0,
            "expected_scale": 1.0,
        },
        {
            "simbolo": "GD30",
            "mercado": "bcba",
            "cantidad": 600.0,
            "costo_unitario": 8_800.0,
            "tipo": "Bonos Globales",
            "tipo_iol": "Bonos Globales",
            "tipo_estandar": "Bonos Globales",
            "moneda": "ARS",
            "valorizado": 5_700_000.0,
            "ultimoPrecio": 9_500.0,
            "expected_scale": 1.0,
        },
        {
            "simbolo": "BPOC7",
            "mercado": "bcba",
            "cantidad": 146.0,
            "costo_unitario": 142_193.15,
            "tipo": "Bonos Provinciales",
            "tipo_iol": "Bonos Provinciales",
            "tipo_estandar": "Bonos Provinciales",
            "moneda": "ARS",
            "valorizado": 200_020.0,
            "ultimoPrecio": 137_000.0,
            "expected_scale": 0.01,
        },
        {
            "simbolo": "S10N5",
            "mercado": "bcba",
            "cantidad": 300.0,
            "costo_unitario": 11_500.0,
            "tipo": "Bonos del Tesoro",
            "tipo_iol": "Bonos del Tesoro",
            "tipo_estandar": "Bonos del Tesoro",
            "moneda": "ARS",
            "valorizado": 3_750_000.0,
            "ultimoPrecio": 12_500.0,
            "expected_scale": 1.0,
        },
    ]

    df_pos = pd.DataFrame(bond_rows)

    def _quote_fn(_mercado: str, _simbolo: str) -> dict[str, object]:
        return {}

    result = calc_rows(_quote_fn, df_pos, exclude_syms=[])

    rows_by_symbol = {row["simbolo"]: row for row in bond_rows}
    expected_last = {
        symbol: data["valorizado"] / (data["cantidad"] * data["expected_scale"])
        for symbol, data in rows_by_symbol.items()
    }

    for symbol, price in expected_last.items():
        match = result.loc[result["simbolo"] == symbol]
        assert not match.empty
        expected_value = rows_by_symbol[symbol]["valorizado"]
        assert math.isclose(match.iloc[0]["valor_actual"], expected_value, rel_tol=1e-6)
        assert math.isclose(match.iloc[0]["ultimo"], price, rel_tol=1e-6)

    audit_section = result.attrs.get("audit", {})
    decisions = {}
    if isinstance(audit_section, dict):
        for item in audit_section.get("scale_decisions", []) or []:
            symbol = item.get("simbolo")
            detected = item.get("scale_detected")
            if symbol is not None and detected is not None:
                decisions[str(symbol)] = float(detected)

    for row in bond_rows:
        symbol = row["simbolo"]
        assert symbol in decisions
        assert math.isclose(decisions[symbol], row["expected_scale"], rel_tol=1e-6)


def test_calc_rows_rescales_bopreal_costs_when_payload_truncated() -> None:
    df_pos = pd.DataFrame(
        [
            {
                "simbolo": "BPOC7",
                "mercado": "bcba",
                "cantidad": 146.0,
                "costo_unitario": 142_193.15,
                # Clasificación neutra para evitar la revaluación forzada y emular el payload truncado.
                "tipo": "Fondos",
                "tipo_iol": "Fondos",
                "tipo_estandar": "Fondos",
                "moneda": "ARS",
                "moneda_origen": "ARS",
                "valorizado": 201_042.0,
                "ultimoPrecio": 1_377.0,
                "provider": "manual",
            }
        ]
    )

    def _quote_fn(_mercado: str, _simbolo: str) -> dict[str, object]:
        return {}

    result = calc_rows(_quote_fn, df_pos, exclude_syms=[])
    row = result.iloc[0]

    expected_ppc = 1_421.9315
    expected_cost = expected_ppc * 146.0
    expected_value = 1_377.0 * 146.0
    expected_pl = expected_value - expected_cost

    assert math.isclose(row["ppc"], expected_ppc, rel_tol=1e-6)
    assert math.isclose(row["costo"], expected_cost, rel_tol=1e-6)
    assert math.isclose(row["valor_actual"], expected_value, rel_tol=1e-6)
    assert math.isclose(row["pl"], expected_pl, rel_tol=1e-6)
    assert math.isclose(row["pl_%"], (expected_pl / expected_cost) * 100.0, rel_tol=1e-6)
