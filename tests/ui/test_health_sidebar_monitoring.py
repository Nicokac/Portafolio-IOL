from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Mapping

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def monitoring_sidebar(monkeypatch: pytest.MonkeyPatch, streamlit_stub):
    import ui.health_sidebar_monitoring as monitoring

    module = importlib.reload(monitoring)
    monkeypatch.setattr(module, "st", streamlit_stub)
    return module


def _sample_payload() -> Mapping[str, object]:
    return {
        "timestamp": "2024-05-18 10:15:00",
        "highlights": [
            {"icon": "✅", "label": "Cotizaciones", "value": "success"},
            {"icon": "⚠️", "label": "FX", "value": "timeout"},
        ],
        "session": {
            "values": {
                "session_id": "abc123",
                "authenticated": True,
                "locale": "es_AR",
            },
            "flags": ["authenticated"],
        },
        "event": "startup.diagnostics",
    }


def test_render_monitoring_sidebar_renders_highlights(monitoring_sidebar, streamlit_stub) -> None:
    payload = dict(_sample_payload())
    monitoring_sidebar.render_monitoring_sidebar(payload)

    markdowns = streamlit_stub.sidebar.markdowns
    assert any("✅ **Cotizaciones:** success" in line for line in markdowns)
    assert any("⚠️ **FX:** timeout" in line for line in markdowns)
    assert any("• locale: es_AR" in line for line in markdowns)
    assert any("✅ authenticated" in line for line in markdowns)

    downloads = streamlit_stub.get_records("download_button")
    assert downloads
    button = downloads[0]
    assert button["label"] == "Descargar diagnóstico"
    assert button["mime"] == "application/json"
    assert "startup.diagnostics" in button["data"]
