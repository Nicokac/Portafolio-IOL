import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from shared import telemetry


@pytest.fixture
def diagnostics_panel(monkeypatch: pytest.MonkeyPatch, streamlit_stub):
    module_name = "ui.panels.system_diagnostics"
    module_path = Path(__file__).resolve().parents[2] / "ui" / "panels" / "system_diagnostics.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "st", streamlit_stub)
    return module


def _build_snapshot() -> SimpleNamespace:
    version = SimpleNamespace(version="1.0.0", build_signature="demo", release_date="2024-01-01")
    endpoints = [
        SimpleNamespace(
            name="predictive_compute",
            average_ms=120.0,
            baseline_ms=110.0,
            degraded=False,
            samples=5,
            last_ms=118.0,
            last_timestamp=1_700_000_000.0,
        )
    ]
    cache = SimpleNamespace(
        hits=10,
        misses=2,
        hit_ratio=0.83,
        last_updated="2024-01-01 00:00:00",
        ttl_hours=1.5,
        remaining_ttl=1800.0,
    )
    keys = [
        SimpleNamespace(
            name="FASTAPI_TOKENS_KEY",
            valid=True,
            is_weak=False,
            fingerprint="abcdef1234",
            detail=None,
        )
    ]
    environment = SimpleNamespace(
        app_env="test",
        timezone="UTC",
        python_version="3.11.0",
        platform="linux",
    )
    return SimpleNamespace(
        generated_at="2024-01-01T00:00:00Z",
        endpoints=endpoints,
        cache=cache,
        keys=keys,
        environment=environment,
        version=version,
    )


def test_system_diagnostics_panel_renders_qa_metrics(
    diagnostics_panel,
    monkeypatch: pytest.MonkeyPatch,
    streamlit_stub,
    tmp_path: Path,
) -> None:
    streamlit_stub.reset()
    qa_path = tmp_path / "qa_metrics.csv"
    qa_rows = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "event_name": "qa_profiler_startup",
            "startup_time_ms": 120.0,
            "ui_render_time_ms": 80.0,
            "cache_load_time_ms": 45.0,
            "auth_latency_ms": 30.0,
            "peak_ram_mb": 150.0,
            "active_threads": 8.0,
            "cached_objects": 3.0,
        },
        {
            "timestamp": "2024-01-01T00:05:00Z",
            "event_name": "qa_profiler_ui",
            "startup_time_ms": 125.0,
            "ui_render_time_ms": 85.0,
            "cache_load_time_ms": 42.0,
            "auth_latency_ms": 28.0,
            "peak_ram_mb": 160.0,
            "active_threads": 9.0,
            "cached_objects": 4.0,
        },
    ]
    pd.DataFrame(qa_rows)[list(telemetry._QA_METRIC_COLUMNS)].to_csv(qa_path, index=False)
    monkeypatch.setattr(telemetry, "_QA_METRICS_FILE", qa_path, raising=False)
    monkeypatch.setattr(diagnostics_panel, "get_system_diagnostics_snapshot", _build_snapshot)

    diagnostics_panel.render_system_diagnostics_panel()

    charts = streamlit_stub.get_records("plotly_chart")
    assert charts, "Expected memory chart to be rendered"
    fig = charts[0]["fig"]
    assert getattr(fig, "data", ()), "Expected chart to include series data"
    first_trace = fig.data[0]
    assert any(value > 0 for value in getattr(first_trace, "y", []))

    tables = [
        record["data"]
        for record in streamlit_stub.get_records("dataframe")
        if isinstance(record.get("data"), pd.DataFrame)
    ]
    qa_table = next(table for table in tables if "Métrica" in table.columns)
    assert set(telemetry._QA_METRIC_COLUMNS[2:]).issubset(set(qa_table["Métrica"]))
