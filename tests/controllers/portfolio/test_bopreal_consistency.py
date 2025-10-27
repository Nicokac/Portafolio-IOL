from __future__ import annotations

import math

import pandas as pd

from application.portfolio_service import BOPREAL_FORCE_FACTOR, calc_rows


def test_calc_rows_rescales_bopreal_last_without_touching_valorizado() -> None:
    qty = 146.0
    ppc = 1_421.9315068493152
    ultimo_truncated = 1_377.0
    valorizado = ultimo_truncated * qty

    df_pos = pd.DataFrame(
        [
            {
                "simbolo": "BPOC7",
                "mercado": "bcba",
                "cantidad": qty,
                "costo_unitario": ppc,
                "tipo": "Fondos",
                "tipo_iol": "Fondos",
                "tipo_estandar": "Fondos",
                "moneda": "ARS",
                "moneda_origen": "ARS",
                "valorizado": valorizado,
                "ultimoPrecio": ultimo_truncated,
                "provider": "manual",
            }
        ]
    )

    result = calc_rows(lambda *_args, **_kwargs: {}, df_pos, exclude_syms=[])

    assert not result.empty
    row = result.iloc[0]

    expected_last = ultimo_truncated * BOPREAL_FORCE_FACTOR
    expected_cost = ppc * qty
    expected_valor_actual = valorizado
    expected_pl = expected_valor_actual - expected_cost
    expected_pl_pct = (expected_pl / expected_cost) * 100.0

    assert math.isclose(row["ultimo"], expected_last, rel_tol=1e-6)
    assert math.isclose(row["valor_actual"], expected_valor_actual, rel_tol=1e-6)
    assert math.isclose(row["costo"], expected_cost, rel_tol=1e-6)
    assert math.isclose(row["pl"], expected_pl, rel_tol=1e-6)
    assert math.isclose(row["pl_%"], expected_pl_pct, rel_tol=1e-6)

    ultimo_precio_column = row.get("ultimoPrecio", math.nan)
    if not pd.isna(ultimo_precio_column):
        assert math.isclose(ultimo_precio_column, expected_last, rel_tol=1e-6)

    audit_section = result.attrs.get("audit", {})
    assert isinstance(audit_section, dict)
    rescale_applied = audit_section.get("bopreal_rescale_applied")
    assert rescale_applied
    if isinstance(rescale_applied, list):
        rescale_applied = rescale_applied[0]
    assert rescale_applied.get("simbolo") == "BPOC7"
    assert rescale_applied.get("factor") == BOPREAL_FORCE_FACTOR
