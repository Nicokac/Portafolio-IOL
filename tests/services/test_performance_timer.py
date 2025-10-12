from __future__ import annotations

import importlib
import importlib
import sys
from pathlib import Path

import pytest


def _reload_timer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    log_path = tmp_path / "performance.log"
    structured_path = tmp_path / "structured.log"
    monkeypatch.setenv("PERFORMANCE_LOG_PATH", str(log_path))
    monkeypatch.setenv("PERFORMANCE_JSON_LOG_PATH", str(structured_path))
    monkeypatch.delenv("PERFORMANCE_TIMER_DISABLE_PSUTIL", raising=False)

    from shared import settings as shared_settings

    shared_settings.settings.REDIS_URL = None
    shared_settings.settings.ENABLE_PROMETHEUS = True
    shared_settings.enable_prometheus = True
    shared_settings.settings.PERFORMANCE_VERBOSE_TEXT_LOG = False
    shared_settings.performance_verbose_text_log = False

    existing = sys.modules.get("services.performance_timer")
    if existing and hasattr(existing, "_shutdown_listener"):
        existing._shutdown_listener()
    sys.modules.pop("services.performance_timer", None)
    module = importlib.import_module("services.performance_timer")
    return module, log_path


def test_performance_timer_logs_elapsed_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    timer, log_path = _reload_timer(tmp_path, monkeypatch)

    with timer.performance_timer("sample_block"):
        pass

    timer._shutdown_listener()

    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "sample_block completado en" in content

    entries = timer.read_recent_entries()
    assert entries
    last = entries[-1]
    assert last.label == "sample_block"
    assert last.duration_s >= 0.0


def test_performance_timer_handles_missing_psutil(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PERFORMANCE_TIMER_DISABLE_PSUTIL", "1")
    timer, log_path = _reload_timer(tmp_path, monkeypatch)

    with timer.performance_timer("no_psutil"):
        pass

    timer._shutdown_listener()

    content = log_path.read_text(encoding="utf-8")
    assert "no_psutil" in content
    # CPU and RAM metrics should be omitted when psutil is disabled.
    last_line = [line for line in content.splitlines() if "no_psutil" in line][-1]
    assert "cpu=" not in last_line.lower()
    assert "ram=" not in last_line.lower()


def test_read_recent_entries_returns_structured_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    timer, _ = _reload_timer(tmp_path, monkeypatch)

    with timer.performance_timer("structured_block", extra={"status": "ok"}):
        pass

    timer._shutdown_listener()

    entries = timer.read_recent_entries(limit=5)
    assert any(entry.label == "structured_block" for entry in entries)
    target = [entry for entry in entries if entry.label == "structured_block"][0]
    assert target.extras.get("status") == "ok"
