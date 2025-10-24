import numpy as np
import pandas as pd
import pytest

from application.portfolio_service import calc_rows, detect_bond_scale_anomalies


@pytest.fixture
def bond_positions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "simbolo": "GD30",
                "mercado": "bcba",
                "tipo": "Bono",
                "cantidad": 10,
                "costo_unitario": 1000.0,
                "moneda": "ARS",
                "moneda_origen": "ARS",
                "ultimoPrecio": 1000.0,
                "valorizado": 10000.0,
            },
            {
                "simbolo": "AL30",
                "mercado": "bcba",
                "tipo": "Bono",
                "cantidad": 100,
                "costo_unitario": 10.0,
                "moneda": "ARS",
                "moneda_origen": "ARS",
                "ultimoPrecio": 950.0,
                "valorizado": np.nan,
            },
            {
                "simbolo": "BPOC7",
                "mercado": "bcba",
                "tipo": "Bono",
                "cantidad": 5,
                "costo_unitario": 9500.0,
                "moneda": "ARS",
                "moneda_origen": "ARS",
                "ultimoPrecio": 9600.0,
                "valorizado": np.nan,
            },
        ]
    )


def _quote_stub(_market: str, _symbol: str):
    return {}


def test_scale_reverts_when_payload_aligned(bond_positions: pd.DataFrame) -> None:
    df_view = calc_rows(_quote_stub, bond_positions, exclude_syms=[])

    gd30 = df_view.loc[df_view["simbolo"] == "GD30"].iloc[0]
    assert pytest.approx(1.0) == gd30["scale"]
    assert pytest.approx(10000.0) == gd30["valor_actual"]
    assert gd30["pricing_source"] == "valorizado"


def test_detects_double_scaling_and_reports_impact(bond_positions: pd.DataFrame) -> None:
    df_view = calc_rows(_quote_stub, bond_positions, exclude_syms=[])

    al30 = df_view.loc[df_view["simbolo"] == "AL30"].iloc[0]
    assert pytest.approx(0.01) == al30["scale"]
    assert pytest.approx(950.0) == al30["valor_actual"]

    bpoc7 = df_view.loc[df_view["simbolo"] == "BPOC7"].iloc[0]
    assert pytest.approx(0.01) == bpoc7["scale"]
    assert pytest.approx(9600.0 * 5 * 0.01) == bpoc7["valor_actual"]

    report, total_impact = detect_bond_scale_anomalies(df_view)
    assert set(report["simbolo"]) == {"GD30", "AL30", "BPOC7"}

    report_indexed = report.set_index("simbolo")
    assert report_indexed.loc["GD30", "diagnostico"] == "correcto"
    assert report_indexed.loc["AL30", "diagnostico"] == "correcto"

    bpoc7_report = report_indexed.loc["BPOC7"]
    assert bpoc7_report["diagnostico"] == "escala duplicada"
    expected_impact = (9600.0 * 5) - (9600.0 * 5 * 0.01)
    assert pytest.approx(expected_impact) == bpoc7_report["impacto_estimado"]
    assert pytest.approx(expected_impact) == total_impact
