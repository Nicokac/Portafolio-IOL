import importlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.time_provider import TimeSnapshot


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

