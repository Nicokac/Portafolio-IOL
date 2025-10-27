from __future__ import annotations

from codecs import BOM_UTF8
from datetime import datetime
from types import SimpleNamespace

import importlib

import pandas as pd

from application.portfolio_service import to_iol_format


def _sample_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simbolo": ["AAPL"],
            "cantidad": [64],
            "pld_%": [2.33],
            "ultimo": [20620.0],
            "ppc": [13755.81],
            "pl_%": [49.9],
            "pl": [439308.0],
            "valor_actual": [1319680.0],
        }
    )


def test_to_iol_format_structure() -> None:
    df = _sample_positions()

    formatted = to_iol_format(df)

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

    assert list(formatted.columns) == expected_columns

    row = formatted.iloc[0]
    assert row["Activo"] == "AAPL"
    assert row["Cantidad"] == "64"
    assert row["Variación diaria"] == "+2,330 %"
    assert row["Último precio"] == "$ 20.620,00"
    assert row["Precio promedio de compra"] == "$ 13.755,81"
    assert row["Rendimiento Porcentaje"] == "49,90%"
    assert row["Rendimiento Monto"] == "$ 439.308,00"
    assert row["Valorizado"] == "$ 1.319.680,00"


def test_export_csv_encoding() -> None:
    df = to_iol_format(_sample_positions())

    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")

    assert csv_bytes.startswith(BOM_UTF8)
    decoded = csv_bytes.decode("utf-8-sig")
    assert decoded.splitlines()[0] == (
        "Activo,Cantidad,Variación diaria,Último precio,Precio promedio de compra,"
        "Rendimiento Porcentaje,Rendimiento Monto,Valorizado"
    )


def test_ui_download_button_exists(monkeypatch, streamlit_stub) -> None:
    module = importlib.import_module("ui.panels.portfolio_comparison")
    module = importlib.reload(module)

    monkeypatch.setattr(module, "st", streamlit_stub)
    streamlit_stub.column_config = SimpleNamespace(
        TextColumn=lambda *args, **kwargs: {"args": args, "kwargs": kwargs}
    )

    tz = module.TimeProvider.timezone()
    fixed_dt = datetime(2025, 10, 25, tzinfo=tz)
    monkeypatch.setattr(module.TimeProvider, "now_datetime", classmethod(lambda cls: fixed_dt))

    streamlit_stub.session_state["portfolio_last_positions"] = _sample_positions()

    module.render_portfolio_comparison_panel()

    download_records = streamlit_stub.get_records("download_button")
    assert download_records
    record = download_records[0]
    assert record["file_name"] == "portafolio_iol_2025-10-25.csv"
    assert record["mime"] == "text/csv"
    assert record["label"] == "💾 Exportar a CSV (Formato IOL)"
    assert isinstance(record["data"], (bytes, bytearray))
    assert record["data"].startswith(BOM_UTF8)

    dataframes = streamlit_stub.get_records("dataframe")
    assert dataframes
    rendered_df = dataframes[0]["data"]
    assert isinstance(rendered_df, pd.DataFrame)
    assert rendered_df.equals(to_iol_format(_sample_positions()))
