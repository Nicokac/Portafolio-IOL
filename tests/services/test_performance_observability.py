from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest
from prometheus_client import generate_latest


def _reload_timer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    enable_prometheus: bool = True,
    app_environment: str = "dev",
):
    log_path = tmp_path / "performance.log"
    structured_path = tmp_path / "structured.log"
    monkeypatch.setenv("PERFORMANCE_LOG_PATH", str(log_path))
    monkeypatch.setenv("PERFORMANCE_JSON_LOG_PATH", str(structured_path))
    monkeypatch.delenv("PERFORMANCE_TIMER_DISABLE_PSUTIL", raising=False)

    from shared import settings as shared_settings

    shared_settings.settings.REDIS_URL = None
    shared_settings.settings.ENABLE_PROMETHEUS = enable_prometheus
    shared_settings.enable_prometheus = enable_prometheus
    shared_settings.settings.PERFORMANCE_VERBOSE_TEXT_LOG = False
    shared_settings.performance_verbose_text_log = False
    shared_settings.settings.app_env = app_environment
    shared_settings.app_env = app_environment

    existing = sys.modules.get("services.performance_timer")
    if existing and hasattr(existing, "_shutdown_listener"):
        existing._shutdown_listener()
    sys.modules.pop("services.performance_timer", None)
    module = importlib.import_module("services.performance_timer")
    return module, log_path, structured_path


def test_performance_timer_writes_structured_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timer, log_path, structured_path = _reload_timer(tmp_path, monkeypatch)

    with timer.performance_timer("structured_block", extra={"workflow": "unit"}):
        pass

    timer._shutdown_listener()

    assert log_path.exists()
    assert structured_path.exists()
    raw_lines = [line for line in structured_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert raw_lines
    payload = json.loads(raw_lines[-1])
    assert payload["label"] == "structured_block"
    assert payload["extras"]["workflow"] == "unit"
    assert payload["success"] is True


def test_store_entry_persists_in_sqlite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timer, _, _ = _reload_timer(tmp_path, monkeypatch, app_environment="prod")
    entry = timer.PerformanceEntry(
        timestamp="2024-01-01T00:00:00Z",
        label="db_store",
        duration_s=1.234,
        cpu_percent=12.5,
        ram_percent=34.5,
        extras={"source": "test"},
        module="tests.performance",
        success=True,
        raw="",
    )

    db_path = tmp_path / "metrics.db"
    monkeypatch.setenv("PERFORMANCE_DB_PATH", str(db_path))
    sys.modules.pop("services.performance_store", None)
    store = importlib.import_module("services.performance_store")
    store.app_env = "prod"
    db_file = store._get_db_path()
    assert db_file.parent.exists()
    assert db_file == db_path
    sqlite3.connect(str(db_path)).close()

    store.store_entry(entry)

    assert db_path.exists()
    conn = sqlite3.connect(db_path)
    try:
        stored = conn.execute(
            "SELECT timestamp, label, duration_s, cpu_pct, mem_pct, extra_json FROM performance_metrics"
        ).fetchone()
    finally:
        conn.close()

    assert stored is not None
    assert stored[1] == "db_store"
    extras = json.loads(stored[-1])
    assert extras["source"] == "test"
    assert extras["module"] == "tests.performance"
    timer._shutdown_listener()


def test_prometheus_metrics_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timer, _, _ = _reload_timer(tmp_path, monkeypatch, enable_prometheus=True)

    with timer.performance_timer("metrics_block"):
        pass

    timer._shutdown_listener()

    assert timer.PROMETHEUS_ENABLED is True
    registry = timer.PROMETHEUS_REGISTRY
    assert registry is not None
    payload = generate_latest(registry).decode("utf-8")
    assert "performance_duration_seconds_count" in payload
    assert 'label="metrics_block"' in payload
