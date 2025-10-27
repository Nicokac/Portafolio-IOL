from __future__ import annotations

from codecs import BOM_UTF8
from datetime import datetime
from types import SimpleNamespace

import importlib

import pandas as pd
import pytest

from application.portfolio_service import _IOL_EXPORT_COLUMNS, to_iol_format


@pytest.fixture(autouse=True)
def _mute_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("application.portfolio_service.log_metric", lambda *_, **__: None)


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


def _sample_bopreal_positions() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "simbolo": ["BPOC7"],
            "cantidad": [146],
            "variacionDiaria": [0.51],
            "pld_%": [0.51],
            "ultimo": [1377.0],
            "ppc": [142_193.15],
            "pl_%": [0.0],
            "pl": [0.0],
            "valor_actual": [201_042.0],
            "moneda_origen": ["ARS"],
            "scale": [0.01],
        }
    )
    df.attrs["audit"] = {"existing": True}
    return df


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


def test_variacion_priority_chain() -> None:
    df = pd.DataFrame(
        {
            "simbolo": ["AAA", "BBB", "CCC", "DDD"],
            "cantidad": [1, 1, 1, 1],
            "variacionDiaria": [0.123, None, None, None],
            "pld_%": [None, 4.56, None, None],
            "chg_pct": [None, None, -7.89, None],
            "ultimo": [10.0, 10.0, 10.0, 10.0],
            "ppc": [10.0, 10.0, 10.0, 10.0],
            "pl_%": [0.0, 0.0, 0.0, 0.0],
            "pl": [0.0, 0.0, 0.0, 0.0],
            "valor_actual": [10.0, 10.0, 10.0, 10.0],
        }
    )

    formatted = to_iol_format(df)
    variations = formatted["Variación diaria"].tolist()

    assert variations == ["+0,123 %", "+4,560 %", "-7,890 %", "+0,000 %"]


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


def test_bopreal_display_rules(monkeypatch) -> None:
    captured: list[tuple[str, dict[str, object] | None]] = []

    def _fake_metric(metric_name: str, context: dict[str, object] | None = None, **_: object) -> None:
        captured.append((metric_name, context))

    monkeypatch.setattr("application.portfolio_service.log_metric", _fake_metric)

    formatted = to_iol_format(_sample_bopreal_positions())

    assert list(formatted.columns) == list(_IOL_EXPORT_COLUMNS)
    row = formatted.iloc[0]
    assert row["Activo"] == "BPOC7"
    assert row["Cantidad"] == "146"
    assert row["Variación diaria"] == "0,510 %"
    assert row["Último precio"] == "$ 137.700,00"
    assert row["Precio promedio de compra"] == "$ 142.193,15"
    assert row["Rendimiento Porcentaje"] == "-3,15%"
    assert row["Rendimiento Monto"] == "-$ 6.560,00"
    assert row["Valorizado"] == "$ 201.042,00"

    assert "audit" in formatted.attrs
    overrides = formatted.attrs["audit"].get("iol_display_overrides")
    assert overrides == ["bopreal_ars"]

    assert captured
    assert captured[0][0] == "comparison_iol.format_version"
    assert captured[0][1] == {"version": "v2"}

    assert len(captured) >= 2
    metric_name, context = captured[1]
    assert metric_name == "comparison_iol.bopreal_rescaled_count"
    assert context is not None
    assert context.get("rows") == 1
    assert context.get("symbols") == ["BPOC7"]

    audit_block = formatted.attrs.get("audit")
    assert isinstance(audit_block, dict)
    assert audit_block.get("existing") is True
    assert audit_block.get("iol_display_overrides") == ["bopreal_ars"]
    assert audit_block.get("bopreal_display_rescaled_rows") == ["BPOC7"]


def test_bopreal_display_scaling_csv_exact_row(monkeypatch) -> None:
    captured: list[tuple[str, dict[str, object] | None]] = []

    def _fake_metric(metric_name: str, context: dict[str, object] | None = None, **_: object) -> None:
        captured.append((metric_name, context))

    monkeypatch.setattr("application.portfolio_service.log_metric", _fake_metric)

    formatted = to_iol_format(_sample_bopreal_positions())

    csv_text = formatted.to_csv(index=False, encoding="utf-8-sig")
    expected_row = (
        "BPOC7,146,\"0,510 %\",\"$ 137.700,00\",\"$ 142.193,15\","
        "\"-3,15%\",\"-$ 6.560,00\",\"$ 201.042,00\"\n"
    )
    assert expected_row in csv_text
    assert "$ 1.377,00" not in csv_text

    assert captured[0][0] == "comparison_iol.format_version"
    assert captured[0][1] == {"version": "v2"}
