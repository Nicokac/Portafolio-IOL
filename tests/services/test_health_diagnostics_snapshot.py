from pathlib import Path
import sys
from types import SimpleNamespace
import logging

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from services import health  # noqa: E402


class FakeTime:
    def __init__(self, start: float) -> None:
        self.value = start

    def time(self) -> float:
        return self.value

    def advance(self, delta: float) -> None:
        self.value += delta


class ExplodingStr:
    def __str__(self) -> str:  # pragma: no cover - used for exception paths
        raise RuntimeError("boom")


@pytest.fixture
def fake_state(monkeypatch):
    state: dict[str, dict] = {}
    monkeypatch.setattr(health, "st", SimpleNamespace(session_state=state))
    return state


def test_record_diagnostics_snapshot_with_valid_payload(monkeypatch, fake_state):
    clock = FakeTime(1_900_000_000.0)
    monkeypatch.setattr(health.time, "time", clock.time)

    health.record_diagnostics_snapshot(
        {
            "status": "ok",
            "latency": 1.23,
            "component": "startup",
            "message": "ready",
            "checks": [
                {"component": "db", "status": "ok", "message": "healthy"},
                {"name": "cache", "status": "stale"},
            ],
        },
        source="engine",
    )

    metrics = health.get_health_metrics()["diagnostics"]
    latest = metrics["latest"]

    assert latest["snapshot"]["status"] == "ok"
    assert latest["snapshot"]["latency"] == pytest.approx(1.23)
    assert latest["snapshot"]["component"] == "startup"
    assert metrics["field_count"] == len(latest["snapshot"])
    assert latest["source"] == "engine"


def test_record_diagnostics_snapshot_with_empty_payload(monkeypatch, fake_state):
    clock = FakeTime(1_900_100_000.0)
    monkeypatch.setattr(health.time, "time", clock.time)

    health.record_diagnostics_snapshot({}, source="ui")

    metrics = health.get_health_metrics()["diagnostics"]
    latest = metrics["latest"]

    assert latest["snapshot"]["status"] == "unknown"
    assert metrics["field_count"] == 1
    assert latest["source"] == "ui"


def test_record_diagnostics_snapshot_warns_on_malformed_payload(monkeypatch, fake_state, caplog):
    clock = FakeTime(1_900_200_000.0)
    monkeypatch.setattr(health.time, "time", clock.time)

    bad_payload = {"status": ExplodingStr()}

    with caplog.at_level(logging.WARNING):
        health.record_diagnostics_snapshot(bad_payload, source="ui")

    assert any(
        "⚠️ No se pudo registrar el diagnóstico de inicio" in record.getMessage()
        for record in caplog.records
    )

    metrics = health.get_health_metrics()["diagnostics"]
    latest = metrics["latest"]

    assert latest["snapshot"]["status"] == "unknown"
    assert latest["source"] == "ui"
    assert metrics["field_count"] == 1
