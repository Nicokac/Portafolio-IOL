import math

import pandas as pd

from application.portfolio_service import calc_rows


def _quote_fn(_mercado: str, _simbolo: str) -> dict[str, object]:
    return {}


def _build_row(**overrides) -> pd.DataFrame:
    base = {
        "simbolo": "BPOC7",
        "mercado": "bcra",
        "tipo": "Bonos",
        "tipo_iol": "Bonos",
        "tipo_estandar": "Bonos",
        "cantidad": 146.0,
        "costo_unitario": 137_600.0,
        "moneda": "ARS",
        "moneda_origen": "ARS",
        "valorizado": 199_669.6,
        "ultimoPrecio": 1_367.6,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_bopreal_forced_revaluation_applies_factor_and_audit() -> None:
    df_pos = _build_row()

    result = calc_rows(_quote_fn, df_pos, exclude_syms=[])
    row = result.iloc[0]

    expected_unit = 1_367.6 * 100.0
    expected_total = expected_unit * 146.0

    assert math.isclose(row["ultimo"], expected_unit, rel_tol=1e-6)
    assert math.isclose(row["valor_actual"], expected_total, rel_tol=1e-6)
    assert row["pricing_source"] == "override_bopreal_forced"

    audit = result.attrs.get("audit", {})
    bopreal_audit = audit.get("bopreal", []) if isinstance(audit, dict) else []
    assert bopreal_audit, "forced override should record bopreal audit entries"

    entry = bopreal_audit[0]
    assert entry.get("forced") is True
    assert entry.get("factor_aplicado") == 100.0
    assert math.isclose(entry.get("ultimo_precio_original"), 1_367.6, rel_tol=1e-6)
    assert math.isclose(entry.get("ultimo_precio_forzado"), expected_unit, rel_tol=1e-6)


def test_bopreal_skip_override_when_price_already_scaled() -> None:
    df_pos = _build_row(ultimoPrecio=137_200.0, valorizado=137_200.0 * 146.0)

    result = calc_rows(_quote_fn, df_pos, exclude_syms=[])
    row = result.iloc[0]

    expected_total = 137_200.0 * 146.0
    assert math.isclose(row["valor_actual"], expected_total, rel_tol=1e-6)
    assert row["pricing_source"] == "ultimoPrecio"

    audit = result.attrs.get("audit", {})
    bopreal_audit = audit.get("bopreal") if isinstance(audit, dict) else None
    assert not bopreal_audit, "no forced override audit entries should be present"


def test_other_bonds_remain_untouched() -> None:
    df_pos = pd.DataFrame(
        [
            {
                "simbolo": "AL30",
                "mercado": "bcba",
                "tipo": "Bonos",
                "tipo_iol": "Bonos",
                "tipo_estandar": "Bonos",
                "cantidad": 10.0,
                "costo_unitario": 9_500.0,
                "moneda": "ARS",
                "moneda_origen": "ARS",
                "valorizado": 98_000.0,
                "ultimoPrecio": 9_800.0,
            }
        ]
    )

    result = calc_rows(_quote_fn, df_pos, exclude_syms=[])
    row = result.iloc[0]

    assert math.isclose(row["valor_actual"], 98_000.0, rel_tol=1e-6)
    assert row["pricing_source"] == "valorizado"

    audit = result.attrs.get("audit", {})
    bopreal_audit = audit.get("bopreal") if isinstance(audit, dict) else None
    assert bopreal_audit in (None, [])
