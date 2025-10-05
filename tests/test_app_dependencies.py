from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "app.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("app", MODULE_PATH)
assert SPEC and SPEC.loader  # pragma: no cover - defensive assertion
app = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("app", app)
SPEC.loader.exec_module(app)


def test_dependency_check_logs_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from services import health

    fake_state: dict[str, object] = {}
    stub_streamlit = SimpleNamespace(session_state=fake_state, stop=lambda: None)
    monkeypatch.setattr(app, "st", stub_streamlit)
    monkeypatch.setattr(health, "st", stub_streamlit)

    original_import = app.importlib.import_module

    def fake_import(name: str, *args, **kwargs):
        if name == "plotly":
            raise ImportError("plotly missing")
        if name == "kaleido":
            return SimpleNamespace()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(app.importlib, "import_module", fake_import)
    monkeypatch.setattr(app, "_ensure_kaleido_runtime_safe", lambda: None)

    caplog.set_level(logging.WARNING, logger="analysis")

    results = app._check_critical_dependencies()

    assert results["plotly"]["status"] == "error"
    assert any("plotly" in record.getMessage().lower() for record in caplog.records)
