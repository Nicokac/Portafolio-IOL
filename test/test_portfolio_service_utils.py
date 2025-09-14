import pandas as pd
import pytest

from application.portfolio_service import (
    map_to_us_ticker,
    classify_symbol,
    scale_for,
    normalize_positions,
    calc_rows,
)


def test_map_to_us_ticker(monkeypatch):
    cfg = {
        "cedear_to_us": {"PAMP": "PAM"},
        "acciones_ar": ["ALUA"],
    }
    monkeypatch.setattr("application.portfolio_service.get_config", lambda: cfg)

    assert map_to_us_ticker("PAMP") == "PAM"
    assert map_to_us_ticker("ALUA") == "ALUA.BA"
    assert map_to_us_ticker("INEXISTENTE") is None


def test_classify_symbol_patterns_and_defaults(monkeypatch):
    cfg = {
        "cedear_to_us": {},
        "etfs": [],
        "acciones_ar": [],
        "fci_symbols": [],
        "classification_patterns": {"Bono": [r"^BON.*"]},
    }
    monkeypatch.setattr("application.portfolio_service.get_config", lambda: cfg)
    monkeypatch.setattr("application.portfolio_service.get_asset_catalog", lambda: {})

    assert classify_symbol("BONAR") == "Bono"
    assert classify_symbol("XYZ") == "CEDEAR"
    assert classify_symbol("ZZ") == "Otro"


def test_scale_for_overrides_and_types(monkeypatch):
    cfg = {"scale_overrides": {"BAD": "abc"}}
    monkeypatch.setattr("application.portfolio_service.get_config", lambda: cfg)

    assert scale_for("BAD", "") == 1.0
    assert scale_for("SOME", "Bono") == 0.01
    assert scale_for("OTHER", "letra") == 0.01


def test_normalize_positions_variants():
    payload_activos = {
        "activos": [
            {"simbolo": "AA", "mercado": "bcba", "cantidad": 1, "costoUnitario": 10}
        ]
    }
    payload_titulos = {
        "titulos": [
            {"titulo": {"simbolo": "BB", "mercado": "nyse", "costoUnitario": 20}, "cantidad": 2}
        ]
    }
    lista_cruda = [
        {"simbolo": "CC", "mercado": "bcba", "cantidad": 3, "costoTotal": 300}
    ]

    df_activos = normalize_positions(payload_activos)
    df_titulos = normalize_positions(payload_titulos)
    df_lista = normalize_positions(lista_cruda)

    assert df_activos.loc[df_activos["simbolo"] == "AA", "costo_unitario"].iloc[0] == 10
    assert df_titulos.loc[df_titulos["simbolo"] == "BB", "costo_unitario"].iloc[0] == 20
    assert df_lista.loc[0, "costo_unitario"] == 100


def test_calc_rows_exclusion_and_values():
    df_pos = pd.DataFrame(
        [
            {"simbolo": "A", "mercado": "bcba", "cantidad": 10, "costo_unitario": 100},
            {"simbolo": "B", "mercado": "bcba", "cantidad": 5, "costo_unitario": 200},
        ]
    )

    def fake_quote(mkt, sym):
        prices = {"A": {"ultimo": 110}, "B": {"ultimo": 210}}
        return prices[sym]

    df = calc_rows(fake_quote, df_pos, ["B"])
    assert list(df["simbolo"]) == ["A"]
    row = df.iloc[0]
    assert isinstance(row["valor_actual"], float)
    assert row["valor_actual"] == pytest.approx(1100.0)
    assert row["costo"] == pytest.approx(1000.0)
