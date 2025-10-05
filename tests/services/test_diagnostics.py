from __future__ import annotations

from pathlib import Path
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def diagnostics_module(monkeypatch: pytest.MonkeyPatch, streamlit_stub):
    import services.diagnostics as diagnostics

    module = importlib.reload(diagnostics)
    monkeypatch.setattr(module, "st", streamlit_stub)
    return module


def test_run_startup_diagnostics_formats_payload(
    diagnostics_module, monkeypatch: pytest.MonkeyPatch, streamlit_stub
) -> None:
    streamlit_stub.session_state.update(
        {
            "session_id": "abc123",
            "authenticated": True,
            "force_login": False,
            "locale": "es_AR",
            "attempts": 3,
            "metadata": {"region": "AR"},
        }
    )

    fake_time = SimpleNamespace(now=lambda: "2024-05-18 10:15:00")
    monkeypatch.setattr(diagnostics_module, "TimeProvider", fake_time)

    health_metrics = {
        "quotes": {"status": "success", "label": "Cotizaciones", "detail": "3 proveedores"},
        "fx_api": {"status": "error", "label": "FX", "detail": "Timeout"},
        "other": "ignored",
    }
    monkeypatch.setattr(diagnostics_module, "get_health_metrics", lambda: health_metrics)

    logger = Mock()
    monkeypatch.setattr(diagnostics_module, "analysis_logger", logger)

    environment_snapshot = {"python": {"version": "3.11.0"}}
    monkeypatch.setattr(
        diagnostics_module,
        "capture_environment_snapshot",
        lambda: environment_snapshot,
    )

    recorder = Mock()
    monkeypatch.setattr(
        diagnostics_module,
        "record_environment_snapshot",
        recorder,
    )

    payload = diagnostics_module.run_startup_diagnostics()

    assert payload["event"] == "startup.diagnostics"
    assert payload["timestamp"] == "2024-05-18 10:15:00"

    session = payload["session"]
    assert session["id"] == "abc123"
    assert "authenticated" in session.get("flags", [])
    assert session["values"]["locale"] == "es_AR"
    assert session["values"]["metadata"] == {"region": "AR"}

    highlights = payload["highlights"]
    assert any(entry["icon"] == "✅" and "Cotizaciones" in entry["label"] for entry in highlights)
    assert any(entry["icon"] == "❌" and entry["label"] == "FX" for entry in highlights)

    assert payload["environment"] == environment_snapshot
    recorder.assert_called_once_with(environment_snapshot)

    logger.info.assert_called_once()
    args, kwargs = logger.info.call_args
    assert args == ("startup.diagnostics",)
    assert kwargs["extra"]["analysis"] == payload
