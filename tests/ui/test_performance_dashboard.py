from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
import pytest


@dataclass
class DummyEntry:
    timestamp: str
    label: str
    duration_s: float
    cpu_percent: float | None
    ram_percent: float | None
    extras: Dict[str, str] = field(default_factory=dict)
    success: bool = True
    module: str = "tests"
    raw: str = ""

    def as_dict(self, include_raw: bool = True) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "timestamp": self.timestamp,
            "label": self.label,
            "duration_s": self.duration_s,
            "cpu_percent": self.cpu_percent,
            "mem_percent": self.ram_percent,
            "module": self.module,
            "success": self.success,
            "extras": dict(self.extras),
        }
        if include_raw and self.raw:
            payload["raw"] = self.raw
        return payload


@pytest.fixture
def performance_dashboard(monkeypatch: pytest.MonkeyPatch, streamlit_stub):
    import sys
    import types

    fake_actions = types.ModuleType("ui.actions")
    fake_actions.render_action_menu = lambda *args, **kwargs: None
    sys.modules.setdefault("ui.actions", fake_actions)

    import ui.tabs.performance_dashboard as dashboard

    module = importlib.reload(dashboard)
    monkeypatch.setattr(module, "st", streamlit_stub)
    return module


def _sample_entries() -> List[DummyEntry]:
    return [
        DummyEntry(
            timestamp="2024-01-01 10:00:00",
            label="load_portfolio",
            duration_s=2.1,
            cpu_percent=30.0,
            ram_percent=40.0,
            extras={"status": "ok"},
            success=True,
        ),
        DummyEntry(
            timestamp="2024-01-01 10:05:00",
            label="load_portfolio",
            duration_s=6.5,
            cpu_percent=85.0,
            ram_percent=55.0,
            extras={"status": "ok"},
            success=True,
        ),
        DummyEntry(
            timestamp="2024-01-01 10:10:00",
            label="predict",
            duration_s=1.5,
            cpu_percent=90.0,
            ram_percent=75.0,
            extras={"note": "spike"},
            success=False,
        ),
        DummyEntry(
            timestamp="2024-01-01 09:55:00",
            label="ui_total_load",
            duration_s=8.0,
            cpu_percent=None,
            ram_percent=None,
            extras={
                "total_ms": "8000",
                "profile_block_total_ms": "6200",
                "streamlit_overhead_ms": "1800",
            },
            success=True,
        ),
    ]


def _collect_metric_records(streamlit_stub) -> List[dict]:
    records: List[dict] = []
    for entry in streamlit_stub.get_records("metric"):
        records.append(entry)
    for columns_entry in streamlit_stub.get_records("columns"):
        for column in columns_entry.get("children", []):
            if column.get("type") != "column":
                continue
            for child in column.get("children", []):
                if child.get("type") == "metric":
                    records.append(child)
    return records


def test_performance_dashboard_renders_metrics(monkeypatch: pytest.MonkeyPatch, streamlit_stub, performance_dashboard) -> None:
    entries = _sample_entries()
    streamlit_stub.reset()
    monkeypatch.setattr(performance_dashboard, "read_recent_entries", lambda limit=200: entries)

    performance_dashboard.render_performance_dashboard_tab(limit=50)

    dataframes = streamlit_stub.get_records("dataframe")
    assert dataframes
    main_table = dataframes[0]["data"]
    assert isinstance(main_table, pd.DataFrame)
    assert "Bloque" in main_table.columns
    assert set(main_table["Bloque"]) == {"load_portfolio", "predict", "ui_total_load"}

    warnings = [entry["text"] for entry in streamlit_stub.get_records("warning")]
    assert any("duración prolongada" in text for text in warnings)

    download_files = {entry["file_name"] for entry in streamlit_stub.get_records("download_button")}
    assert download_files == {
        "performance_metrics.csv",
        "performance_metrics.json",
        "performance_sparkline.csv",
    }

    line_charts = streamlit_stub.get_records("line_chart")
    assert len(line_charts) >= 1

    metric_records = _collect_metric_records(streamlit_stub)
    assert metric_records
    assert any(record["label"] == "Duración última (s)" for record in metric_records)
    duration_metric = next(
        record for record in metric_records if record["label"] == "Duración última (s)"
    )
    assert duration_metric.get("chart_color") == list(performance_dashboard._GRADIENT_POSITIVE)

    ui_total_metric = next(
        (record for record in metric_records if record["label"] == "Total carga UI"),
        None,
    )
    assert ui_total_metric is not None
    overhead_metric = next(
        (record for record in metric_records if record["label"] == "Overhead Streamlit"),
        None,
    )
    assert overhead_metric is not None
    assert str(overhead_metric.get("value", "")).endswith(" ms")

    percentiles_table = None
    for record in dataframes:
        df = record["data"]
        if isinstance(df, pd.DataFrame) and {"P50 (s)", "P95 (s)", "P99 (s)"}.issubset(df.columns):
            percentiles_table = df
            break
    assert percentiles_table is not None

    captions = [entry["text"] for entry in streamlit_stub.get_records("caption")]
    assert any("Archivo de log" in text for text in captions)
    assert any("Modo seleccionado" in text for text in captions)

    markdowns = [entry["text"] for entry in streamlit_stub.get_records("markdown")]
    assert any("staticPlot=True" in text for text in markdowns)


def test_performance_dashboard_historical_mode(
    monkeypatch: pytest.MonkeyPatch, streamlit_stub, performance_dashboard
) -> None:
    entries = _sample_entries()
    streamlit_stub.reset()
    streamlit_stub._toggle_returns[performance_dashboard._VIEW_MODE_TOGGLE_KEY] = True
    monkeypatch.setattr(performance_dashboard, "read_recent_entries", lambda limit=200: entries)

    performance_dashboard.render_performance_dashboard_tab(limit=50)

    assert (
        streamlit_stub.session_state[performance_dashboard._VIEW_MODE_STATE_KEY]
        == "historical"
    )

    metric_records = _collect_metric_records(streamlit_stub)
    duration_metric = next(
        record for record in metric_records if record["label"] == "Duración promedio (s)"
    )
    assert duration_metric.get("chart_color") == list(performance_dashboard._GRADIENT_POSITIVE)

    sparkline_download = next(
        record
        for record in streamlit_stub.get_records("download_button")
        if record["file_name"] == "performance_sparkline.csv"
    )
    csv_payload = sparkline_download["data"].decode("utf-8")
    assert "view_mode" in csv_payload
    assert "historical" in csv_payload


def test_performance_dashboard_keyword_filter(monkeypatch: pytest.MonkeyPatch, streamlit_stub, performance_dashboard) -> None:
    entries = _sample_entries()
    streamlit_stub.reset()
    monkeypatch.setattr(performance_dashboard, "read_recent_entries", lambda limit=200: entries)

    original_text_input = streamlit_stub.text_input

    def fake_text_input(label: str, *, key: str | None = None, value: str | None = None, help: str | None = None) -> str:
        return original_text_input(label, key=key, value="spike", help=help)

    monkeypatch.setattr(streamlit_stub, "text_input", fake_text_input, raising=False)

    performance_dashboard.render_performance_dashboard_tab(limit=50)

    dataframes = streamlit_stub.get_records("dataframe")
    assert dataframes
    main_table = dataframes[0]["data"]
    assert set(main_table["Bloque"]) == {"predict"}

    warnings = [entry["text"] for entry in streamlit_stub.get_records("warning")]
    assert warnings  # Alert should remain for the filtered entry due to CPU/mem thresholds
