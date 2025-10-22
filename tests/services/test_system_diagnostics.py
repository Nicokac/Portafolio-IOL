from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timezone

import pytest
from cryptography.fernet import Fernet

from application.predictive_service import PredictiveSnapshot
from services.performance_metrics import MetricSummary


@pytest.fixture()
def diagnostics_module(monkeypatch: pytest.MonkeyPatch, tmp_path):
    module_name = "services.system_diagnostics"
    monkeypatch.setenv("SYSTEM_DIAGNOSTICS_LOG_PATH", str(tmp_path / "system_diag.log"))
    monkeypatch.setenv("FASTAPI_TOKENS_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("IOL_TOKENS_KEY", Fernet.generate_key().decode())
    if module_name in sys.modules:
        del sys.modules[module_name]
    module = importlib.import_module(module_name)
    return module


def _summary(name: str, average: float, samples: int = 5) -> MetricSummary:
    return MetricSummary(
        name=name,
        samples=samples,
        average_ms=average,
        last_ms=average,
        average_memory_kb=None,
        last_memory_kb=None,
        last_timestamp=1700000000.0,
        version="test",
    )


def test_run_diagnostics_detects_degradation_and_logs(
    diagnostics_module, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    module = diagnostics_module

    monkeypatch.setattr(
        module.TimeProvider,
        "now_datetime",
        classmethod(lambda cls: datetime(2024, 1, 1, tzinfo=timezone.utc)),
    )
    cache_snapshot = PredictiveSnapshot(
        hits=12,
        misses=3,
        last_updated="2024-01-01T00:00:00Z",
        ttl_hours=6.0,
        remaining_ttl=7200.0,
    )
    monkeypatch.setattr(module, "get_cache_stats", lambda: cache_snapshot)

    monkeypatch.setattr(
        module,
        "get_recent_metrics",
        lambda: [
            _summary("predictive_compute", 120.0),
            _summary("apply_filters", 90.0),
        ],
    )
    first_snapshot = module.run_system_diagnostics_once()
    assert any(entry.name == "predictive_compute" for entry in first_snapshot.endpoints)
    assert not any(entry.degraded for entry in first_snapshot.endpoints)

    monkeypatch.setattr(
        module,
        "get_recent_metrics",
        lambda: [
            _summary("predictive_compute", 260.0),
            _summary("apply_filters", 190.0),
        ],
    )
    second_snapshot = module.run_system_diagnostics_once()
    degraded = {entry.name for entry in second_snapshot.endpoints if entry.degraded}
    assert "predictive_compute" in degraded

    log_path = tmp_path / "system_diag.log"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "Diagnostics log should contain entries"
    payload = json.loads(lines[-1].split("] ", 1)[-1])
    assert payload["endpoints"], "Log payload should include endpoints"


def test_key_status_marks_duplicates(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    key = Fernet.generate_key().decode()
    module_name = "services.system_diagnostics"
    monkeypatch.setenv("SYSTEM_DIAGNOSTICS_LOG_PATH", str(tmp_path / "diag.log"))
    monkeypatch.setenv("FASTAPI_TOKENS_KEY", key)
    monkeypatch.setenv("IOL_TOKENS_KEY", key)
    if module_name in sys.modules:
        del sys.modules[module_name]
    module = importlib.import_module(module_name)

    monkeypatch.setattr(module, "get_recent_metrics", lambda: [])
    monkeypatch.setattr(module, "get_cache_stats", lambda: None)
    snapshot = module.run_system_diagnostics_once()

    assert snapshot.keys
    assert all(not status.valid for status in snapshot.keys)
    assert any("no pueden coincidir" in (status.detail or "") for status in snapshot.keys)
