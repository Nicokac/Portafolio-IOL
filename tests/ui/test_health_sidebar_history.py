import importlib
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from shared.time_provider import TimeSnapshot

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def history_module(monkeypatch, streamlit_stub):
    streamlit_stub.reset()

    def _sidebar_subheader(text):
        entry = {"type": "subheader", "text": str(text)}
        streamlit_stub.sidebar.elements.append(entry)

    def _sidebar_plotly(fig, **kwargs):
        entry = {"type": "plotly_chart", "figure": fig, "kwargs": kwargs}
        streamlit_stub.sidebar.elements.append(entry)
        streamlit_stub._record("plotly_chart", fig=fig, kwargs=kwargs)

    streamlit_stub.sidebar.subheader = _sidebar_subheader
    streamlit_stub.sidebar.plotly_chart = _sidebar_plotly

    module = importlib.reload(importlib.import_module("ui.health_sidebar_history"))
    monkeypatch.setattr(module, "st", streamlit_stub)

    snapshots = {
        111.0: TimeSnapshot("2024-05-18 10:00:00", datetime(2024, 5, 18, 10, 0, 0)),
        222.0: TimeSnapshot("2024-05-18 10:01:00", datetime(2024, 5, 18, 10, 1, 0)),
    }

    def _from_timestamp(ts):
        try:
            value = float(ts)
        except (TypeError, ValueError):
            return None
        return snapshots.get(value)

    monkeypatch.setattr(module, "TimeProvider", SimpleNamespace(from_timestamp=_from_timestamp))

    class DummyFigure:
        def __init__(self) -> None:
            self.traces = []
            self.layout_updates = []

        def add_trace(self, trace):
            self.traces.append(trace)

        def update_layout(self, **kwargs):
            self.layout_updates.append(kwargs)

    monkeypatch.setattr(module, "go", SimpleNamespace(Figure=DummyFigure, Bar=lambda **kw: ("bar", kw)))

    streamlit_stub.reset()
    return module


def test_history_sidebar_renders_chart_and_annotations(history_module, streamlit_stub):
    metrics = {
        "history": [
            {
                "ts": 111.0,
                "elapsed_ms": 320.0,
                "status": "success",
                "detail": "Primario",
                "environment": ["CI"],
            },
            {
                "ts": 222.0,
                "elapsed_ms": 540.0,
                "status": "error",
                "detail": "Fallback",
                "environment": ["CI", "Fallback"],
            },
        ],
        "environment": ["CI", "Fallback"],
        "last_error": {"ts": 222.0, "label": "Fallback detectado"},
    }

    history_module.render_history_sidebar(metrics)

    plot_calls = streamlit_stub.get_records("plotly_chart")
    assert plot_calls, "Expected history chart to be rendered"

    markdowns = [entry["text"] for entry in streamlit_stub.sidebar.elements if entry["type"] == "markdown"]
    assert any("Ambiente:" in text and "`CI`" in text for text in markdowns)
    assert any("Último error:" in text and "Fallback detectado" in text for text in markdowns)
    assert any("2024-05-18 10:01:00" in text and "Fallback" in text for text in markdowns)


def _render(module, metrics: dict[str, Any]) -> None:
    module.st.session_state["health_metrics"] = metrics
    module.render_health_sidebar()


@pytest.fixture
def health_sidebar(streamlit_stub, monkeypatch: pytest.MonkeyPatch):
    import ui.health_sidebar as health_sidebar_module

    module = importlib.reload(health_sidebar_module)
    monkeypatch.setattr(
        module,
        "get_health_metrics",
        lambda: module.st.session_state.get("health_metrics", {}),
    )
    return module


def test_recent_stats_render_charts_and_badge(health_sidebar, streamlit_stub) -> None:
    metrics = {
        "environment_snapshot": {
            "python_version": "3.11.6",
            "streamlit_version": "1.32.0",
            "runtime": "Local",
            "ts": 1700000000.0,
        },
        "authentication": {},
        "iol_refresh": {},
        "yfinance": {},
        "snapshot": {},
        "fx_api": {
            "stats": {"latency": {"samples": [110.0, 95.0, 102.0]}},
        },
        "fx_cache": {
            "stats": {"age": {"samples": [1.0, 1.5, 0.5]}},
        },
        "portfolio": {
            "stats": {"latency": {"samples": [200.0, 180.0, 220.0]}},
        },
        "quotes": {
            "stats": {"latency": {"samples": [320.0, 300.0]}},
        },
        "session_monitoring": {
            "http_errors": {
                "count": 2,
                "last": {
                    "status_code": 502,
                    "method": "get",
                    "url": "/api/portfolio",
                    "detail": "Bad gateway",
                    "ts": 1699999999.0,
                },
            }
        },
    }

    _render(health_sidebar, metrics)

    assert any("🐍" in markdown and "Streamlit" in markdown for markdown in streamlit_stub.sidebar.markdowns), (
        "Expected environment badge in sidebar"
    )

    line_charts = [element for element in streamlit_stub.sidebar.elements if element["type"] == "line_chart"]
    assert line_charts, "Expected a line chart with latency samples"
    assert isinstance(line_charts[0]["data"], pd.DataFrame)

    area_charts = [element for element in streamlit_stub.sidebar.elements if element["type"] == "area_chart"]
    assert area_charts, "Expected an area chart with cache age samples"
    assert isinstance(area_charts[0]["data"], pd.DataFrame)

    assert any("Último error HTTP" in markdown for markdown in streamlit_stub.sidebar.markdowns), (
        "Expected last HTTP error note"
    )


def test_recent_stats_section_handles_missing_samples(health_sidebar, streamlit_stub) -> None:
    metrics: dict[str, Any] = {
        "authentication": {},
        "iol_refresh": {},
        "yfinance": {},
        "snapshot": {},
        "session_monitoring": {},
    }

    _render(health_sidebar, metrics)

    recent_section = [markdown for markdown in streamlit_stub.sidebar.markdowns if "Estadísticas recientes" in markdown]
    assert recent_section, "Section header should be rendered"

    assert "_Sin estadísticas recientes._" in streamlit_stub.sidebar.markdowns
    assert not [
        element for element in streamlit_stub.sidebar.elements if element["type"] in {"line_chart", "area_chart"}
    ]
