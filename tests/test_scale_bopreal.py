import math

import pandas as pd

from application.portfolio_service import calc_rows


def _quote_fn(_mercado: str, _simbolo: str) -> dict[str, object]:
    return {}


def _build_bopreal_row(**overrides) -> pd.DataFrame:
    base = {
        "simbolo": "BPOC7",
        "mercado": "bcri",
        "tipo": "Bonos",
        "tipo_iol": "Bonos",
        "tipo_estandar": "Bonos",
        "cantidad": 146.0,
        "costo_unitario": 100.0,
        "moneda": "ARS",
        "moneda_origen": "ARS",
        "valorizado": 199_669.6,
        "ultimoPrecio": 1_367.6,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_bopreal_forced_revaluation_uses_ultimo():
    df_pos = _build_bopreal_row()

    result = calc_rows(_quote_fn, df_pos, exclude_syms=[])

    assert math.isclose(result.iloc[0]["ultimo"], 1_367.6, rel_tol=1e-6)
    expected_total = 1_367.6 * 146.0
    assert math.isclose(result.iloc[0]["valor_actual"], expected_total, rel_tol=1e-6)
    assert result.iloc[0]["pricing_source"] == "ultimoPrecio"

    audit = result.attrs.get("audit", {})
    decisions = audit.get("scale_decisions", []) if isinstance(audit, dict) else []
    assert any(
        isinstance(entry, dict)
        and entry.get("simbolo") == "BPOC7"
        and entry.get("reason") == "override_bopreal_ars_forced_revaluation"
        for entry in decisions
    )


def test_bopreal_forced_revaluation_rescales_valorizado_when_ultimo_missing():
    df_pos = _build_bopreal_row(ultimoPrecio=float("nan"))

    result = calc_rows(_quote_fn, df_pos, exclude_syms=[])

    expected_total = 199_669.6 * 100.0
    assert math.isclose(result.iloc[0]["valor_actual"], expected_total, rel_tol=1e-6)
    assert math.isclose(result.iloc[0]["ultimo"], expected_total / 146.0, rel_tol=1e-6)
    assert result.iloc[0]["pricing_source"] == "valorizado_rescaled"

    audit = result.attrs.get("audit", {})
    decisions = audit.get("scale_decisions", []) if isinstance(audit, dict) else []
    assert any(
        isinstance(entry, dict)
        and entry.get("simbolo") == "BPOC7"
        and entry.get("reason") == "override_bopreal_ars_forced_revaluation"
        for entry in decisions
    )
