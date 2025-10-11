from __future__ import annotations

import time

import pytest

from services import performance_metrics
from services import update_checker
from shared.version import __version__


@pytest.fixture(autouse=True)
def _reset_metrics() -> None:
    performance_metrics.reset_metrics()
    yield
    performance_metrics.reset_metrics()


def test_measure_execution_persists_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    log_path = tmp_path / "update_log.json"
    monkeypatch.setattr(update_checker, "LOG_FILE", str(log_path), raising=False)

    with performance_metrics.measure_execution("sample_task"):
        time.sleep(0.001)

    metrics = performance_metrics.get_recent_metrics()
    assert metrics, "Se esperan métricas registradas para sample_task"
    summary = metrics[0]
    assert summary.name == "sample_task"
    assert summary.samples == 1
    assert summary.last_ms >= 0.0
    if summary.average_memory_kb is not None:
        assert summary.average_memory_kb >= 0.0

    history = update_checker.get_update_history()
    assert history, "Se espera que el log de actualizaciones reciba eventos de performance"
    event = history[-1]
    assert event["event"] == "perf:sample_task"
    assert event["version"] == __version__
    assert "duration=" in event["status"]

    csv_payload = performance_metrics.export_metrics_csv()
    assert "sample_task" in csv_payload
    assert "average_ms" in csv_payload.splitlines()[0]
