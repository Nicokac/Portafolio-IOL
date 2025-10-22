import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def system_status_panel(monkeypatch: pytest.MonkeyPatch, streamlit_stub):
    module_name = "ui.panels.system_status"
    module_path = Path(__file__).resolve().parents[2] / "ui" / "panels" / "system_status.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "st", streamlit_stub)
    return module


def test_system_status_panel_renders_tabs_and_metrics(system_status_panel, streamlit_stub) -> None:
    streamlit_stub.reset()
    streamlit_stub.session_state["prometheus_metrics"] = {
        "uptime_seconds": 7200.0,
        "auth_refresh_total": 5.0,
        "cache_hit_ratio": 0.82,
        "performance_duration_seconds_count": 120,
        "auth_failures_total": 2,
        "cache_evictions_total": 6,
    }
    streamlit_stub.session_state["auth_token_claims"] = {
        "sub": "demo_user",
        "iat": 1_700_000_000,
        "exp": 1_700_000_900,
        "session_id": "abc123",
        "ttl": 900,
    }
    streamlit_stub.session_state["auth_token_refreshed_at"] = "2024-01-01 12:00:00"

    system_status_panel.render_system_status_panel()

    headers = [entry["text"] for entry in streamlit_stub.get_records("header")]
    assert "🔍 Estado del Sistema" in headers

    markdowns = [entry["text"] for entry in streamlit_stub.get_records("markdown")]
    assert any("Documentación operativa" in text for text in markdowns)

    tabs = streamlit_stub.get_records("tabs")
    assert tabs
    assert tabs[0]["labels"] == ["⚡ Performance", "🔐 Seguridad", "🗃️ Caché"]

    columns = streamlit_stub.get_records("columns")
    assert columns
    summary_columns = columns[0]["children"]
    summary_labels = [
        child["label"]
        for column in summary_columns
        for child in column.get("children", [])
        if child.get("type") == "metric"
    ]
    assert {"🕒 Uptime", "🔁 Refresh tokens", "📦 Hit ratio caché"}.issubset(summary_labels)

    dataframes = streamlit_stub.get_records("dataframe")
    assert len(dataframes) >= 3
    tables = [record["data"] for record in dataframes if isinstance(record["data"], pd.DataFrame)]
    assert any("auth_failures_total" in table["Métrica"].values for table in tables)
    assert any("cache_evictions_total" in table["Métrica"].values for table in tables)

    token_metrics = [
        child["label"]
        for column in columns[1]["children"]
        for child in column.get("children", [])
        if child.get("type") == "metric"
    ]
    assert {"Usuario", "TTL", "TTL restante"}.issubset(token_metrics)

    buttons = streamlit_stub.get_records("button")
    assert any(entry["label"] == "🔄 Refrescar token" for entry in buttons)
