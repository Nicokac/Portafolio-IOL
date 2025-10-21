import importlib
from types import SimpleNamespace

import pytest


def test_health_snapshots_are_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    health = importlib.import_module("services.health")
    session_adapter = importlib.import_module("services.health.session_adapter")

    state: dict[str, object] = {}
    st_stub = SimpleNamespace(session_state=state)
    monkeypatch.setattr(session_adapter, "st", st_stub, raising=False)
    monkeypatch.setattr(health, "st", st_stub, raising=False)

    state.clear()

    health.record_diagnostics_snapshot(
        {
            "status": "ok",
            "component": "bootstrap",
            "latency": 42.5,
            "snapshot": {
                "python": {"version": "3.11.0"},
                "env": "test",
            },
        },
        source="startup",
    )
    health.record_snapshot_event(
        kind="export",
        status="success",
        action="save",
        storage_id="snap-001",
        detail="Completed",
        backend={"name": "local", "status": "ok"},
    )

    metrics = health.get_health_metrics()

    diagnostics = metrics["diagnostics"]
    latest = diagnostics["latest"]
    assert latest["source"] == "startup"
    assert latest["snapshot"]["python"]["version"] == "3.11.0"
    assert diagnostics["field_count"] == 2

    event = metrics["snapshot_event"]
    assert event["kind"] == "export"
    assert event["status"] == "success"
    assert event["action"] == "save"
    assert event["storage_id"] == "snap-001"
    assert event["backend"]["name"] == "local"
    assert isinstance(event["ts"], float)
