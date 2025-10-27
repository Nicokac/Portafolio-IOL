from __future__ import annotations

import importlib
from datetime import datetime
from types import SimpleNamespace

import pandas as pd


def _sample_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simbolo": ["AAPL"],
            "cantidad": [64],
            "pld_%": [2.33],
            "chg_pct": [2.33],
            "ultimo": [20620.0],
            "ppc": [13755.81],
            "pl_%": [49.9],
            "pl": [439308.0],
            "valor_actual": [1319680.0],
        }
    )


def _bopreal_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simbolo": ["BPOC7"],
            "cantidad": [146],
            "pld_%": [0.51],
            "ultimo": [1377.0],
            "ppc": [142_193.15],
            "pl_%": [0.0],
            "pl": [0.0],
            "valor_actual": [1.0],
            "moneda_origen": ["ARS"],
        }
    )


def _reload_panel(monkeypatch, streamlit_stub, df: pd.DataFrame | None = None):
    module = importlib.import_module("ui.panels.portfolio_comparison")
    module = importlib.reload(module)
    monkeypatch.setattr(module, "st", streamlit_stub)
    tz = module.TimeProvider.timezone()
    fixed_dt = datetime(2025, 10, 25, tzinfo=tz)
    monkeypatch.setattr(
        module.TimeProvider,
        "now_datetime",
        classmethod(lambda cls: fixed_dt),
    )
    dataset = df if df is not None else _sample_positions()
    streamlit_stub.session_state["portfolio_last_positions"] = dataset
    streamlit_stub.session_state["portfolio_last_viewmodel"] = SimpleNamespace(positions=dataset)
    return module


def test_render_panel_without_alignment(monkeypatch, streamlit_stub) -> None:
    module = _reload_panel(monkeypatch, streamlit_stub)

    def text_column(label, *, width=None):
        return {"label": label, "width": width}

    streamlit_stub.column_config = SimpleNamespace(TextColumn=text_column)

    module.render_portfolio_comparison_panel()

    dataframes = streamlit_stub.get_records("dataframe")
    assert dataframes
    column_config = dataframes[0]["column_config"]
    assert column_config is not None
    assert list(column_config) == [
        "Activo",
        "Cantidad",
        "Variación diaria",
        "Último precio",
        "Precio promedio de compra",
        "Rendimiento Porcentaje",
        "Rendimiento Monto",
        "Valorizado",
    ]


def test_column_config_structure(monkeypatch, streamlit_stub) -> None:
    module = _reload_panel(monkeypatch, streamlit_stub)

    captured: dict[str, dict[str, object]] = {}

    def text_column(label, *, width=None, alignment="left"):
        captured[label] = {"width": width, "alignment": alignment}
        return {"label": label, "width": width, "alignment": alignment}

    streamlit_stub.column_config = SimpleNamespace(TextColumn=text_column)

    module.render_portfolio_comparison_panel()

    expected_columns = [
        "Activo",
        "Cantidad",
        "Variación diaria",
        "Último precio",
        "Precio promedio de compra",
        "Rendimiento Porcentaje",
        "Rendimiento Monto",
        "Valorizado",
    ]

    dataframes = streamlit_stub.get_records("dataframe")
    assert dataframes
    column_config = dataframes[0]["column_config"]
    assert column_config is not None
    assert list(column_config) == expected_columns

    expected_alignment = {
        "Activo": ("medium", "left"),
        "Cantidad": ("small", "right"),
        "Variación diaria": ("small", "right"),
        "Último precio": (None, "right"),
        "Precio promedio de compra": (None, "right"),
        "Rendimiento Porcentaje": ("small", "right"),
        "Rendimiento Monto": (None, "right"),
        "Valorizado": (None, "right"),
    }

    assert captured.keys() == column_config.keys()
    for column_name, (width, alignment) in expected_alignment.items():
        assert captured[column_name]["width"] == width
        assert captured[column_name]["alignment"] == alignment


def test_bopreal_row_matches_iol(monkeypatch, streamlit_stub) -> None:
    module = _reload_panel(monkeypatch, streamlit_stub, _bopreal_positions())

    streamlit_stub.column_config = SimpleNamespace(TextColumn=lambda *args, **kwargs: None)

    module.render_portfolio_comparison_panel()

    dataframes = streamlit_stub.get_records("dataframe")
    assert dataframes
    df_rendered = dataframes[0]["data"]
    assert isinstance(df_rendered, pd.DataFrame)
    assert list(df_rendered.columns) == [
        "Activo",
        "Cantidad",
        "Variación diaria",
        "Último precio",
        "Precio promedio de compra",
        "Rendimiento Porcentaje",
        "Rendimiento Monto",
        "Valorizado",
    ]

    row = df_rendered.iloc[0]
    assert row["Activo"] == "BPOC7"
    assert row["Variación diaria"] == "0,510 %"
    assert row["Último precio"] == "$ 137.700,00"
    assert row["Precio promedio de compra"] == "$ 142.193,15"
    assert row["Rendimiento Porcentaje"] == "-3,15%"
    assert row["Rendimiento Monto"] == "-$ 6.560,00"
    assert row["Valorizado"] == "$ 201.042,00"


def test_csv_contains_bopreal_expected_strings(monkeypatch, streamlit_stub) -> None:
    module = _reload_panel(monkeypatch, streamlit_stub, _bopreal_positions())

    streamlit_stub.column_config = SimpleNamespace(TextColumn=lambda *args, **kwargs: None)

    module.render_portfolio_comparison_panel()

    downloads = streamlit_stub.get_records("download_button")
    assert downloads
    payload = downloads[0]["data"]
    assert isinstance(payload, (bytes, bytearray))
    text = payload.decode("utf-8-sig")

    assert "BPOC7" in text
    assert "$ 137.700,00" in text
    assert "$ 142.193,15" in text
    assert "-$ 6.560,00" in text
    assert "$ 201.042,00" in text
