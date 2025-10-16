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
    ]


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
    assert set(main_table["Bloque"]) == {"load_portfolio", "predict"}

    warnings = [entry["text"] for entry in streamlit_stub.get_records("warning")]
    assert any("duración prolongada" in text for text in warnings)

    download_files = {entry["file_name"] for entry in streamlit_stub.get_records("download_button")}
    assert download_files == {"performance_metrics.csv", "performance_metrics.json"}

    line_charts = streamlit_stub.get_records("line_chart")
    assert len(line_charts) >= 1

    metric_records = streamlit_stub.get_records("metric")
    assert metric_records
    assert any(record["label"] == "Duración última (s)" for record in metric_records)

    percentiles_table = None
    for record in dataframes:
        df = record["data"]
        if isinstance(df, pd.DataFrame) and {"P50 (s)", "P95 (s)", "P99 (s)"}.issubset(df.columns):
            percentiles_table = df
            break
    assert percentiles_table is not None

    captions = [entry["text"] for entry in streamlit_stub.get_records("caption")]
    assert any("Archivo de log" in text for text in captions)


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
