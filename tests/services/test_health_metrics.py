from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from services import health  # noqa: E402
from tests.fixtures.time import FakeTime


def test_quote_provider_summary_handles_mixed_data(monkeypatch):
    fake_state: dict[str, dict] = {}
    monkeypatch.setattr(health, "st", SimpleNamespace(session_state=fake_state))

    store = health._store()
    store["quote_providers"] = {
        "alpha": {
            "count": 3,
            "ok_count": 2,
            "stale_count": 1,
            "elapsed_history": [0.1, 0.2, 0.3],
        },
        "beta": {
            "count": 5,
            "stale_count": 0,
        },
        "gamma": {
            "count": 4,
            "ok_count": 0,
        },
    }
    store[health._QUOTE_RATE_LIMIT_KEY] = {
        "alpha": {
            "count": 1,
            "wait_total": 0.1,
            "wait_last": 0.05,
        }
    }

    summary = health.get_health_metrics()["quote_providers"]

    assert summary["total"] == 12
    assert summary["ok_total"] == 2

    providers = {entry["provider"]: entry for entry in summary["providers"]}
    alpha = providers["alpha"]
    assert alpha["ok_count"] == 2
    assert alpha["ok_ratio"] == pytest.approx(2 / 3)
    assert "rate_limit_count" in alpha


def test_quote_provider_summary_returns_dict(monkeypatch):
    fake_state: dict[str, dict] = {}
    monkeypatch.setattr(health, "st", SimpleNamespace(session_state=fake_state))

    store = health._store()
    store["quote_providers"] = {"alpha": {"count": 1, "ok_count": 1}}

    summary = health.get_health_metrics()["quote_providers"]

    assert isinstance(summary, dict)
    assert summary["providers"]


def test_record_snapshot_event_stores_backend_details(monkeypatch):
    fake_state: dict[str, dict] = {}
    monkeypatch.setattr(health, "st", SimpleNamespace(session_state=fake_state))

    health.record_snapshot_event(
        kind="portfolio",
        status="saved",
        action="save",
        storage_id="abc123",
        backend={"name": "json", "path": "/tmp/snapshots.json"},
    )

    event = health.get_health_metrics()["snapshot_event"]

    assert event["kind"] == "portfolio"
    assert event["status"] == "saved"
    assert event["action"] == "save"
    assert event["storage_id"] == "abc123"
    assert event["backend"]["name"] == "json"


def test_session_monitoring_metrics(monkeypatch):
    fake_state: dict[str, dict] = {}
    monkeypatch.setattr(health, "st", SimpleNamespace(session_state=fake_state))

    clock = FakeTime(1_700_000_000.0)
    monkeypatch.setattr(health.time, "time", clock.time)

    health.record_session_started("sess-1", metadata={"user": "alice", "empty": "  "})
    clock.sleep(2.0)
    health.record_login_to_render(1.5, session_id="sess-1")
    clock.sleep(1.0)
    health.record_http_error(500, method="GET", url="/api", detail=" boom ")
    clock.sleep(5.0)

    metrics = health.get_health_metrics()
    monitoring = metrics["session_monitoring"]

    active = monitoring["active_sessions"]
    assert active["count"] == 1
    assert active["sessions"][0]["session_id"] == "sess-1"
    assert active["sessions"][0]["metadata"]["user"] == "alice"
    assert active["freshness"]["is_fresh"] is True

    login_stats = monitoring["login_to_render"]
    assert login_stats["count"] == 1
    assert login_stats["avg"] == pytest.approx(1.5)
    assert login_stats["last"]["session_id"] == "sess-1"
    assert login_stats["freshness"]["is_fresh"] is True

    http_errors = monitoring["http_errors"]
    assert http_errors["count"] == 1
    assert http_errors["last"]["status_code"] == 500
    assert http_errors["last"]["detail"] == "boom"
    assert http_errors["freshness"]["is_fresh"] is True


def test_diagnostics_snapshot_summary(monkeypatch):
    fake_state: dict[str, dict] = {}
    monkeypatch.setattr(health, "st", SimpleNamespace(session_state=fake_state))

    clock = FakeTime(1_800_000_000.0)
    monkeypatch.setattr(health.time, "time", clock.time)

    health.record_diagnostics_snapshot({"status": "ok", " empty ": ""}, source="engine")
    clock.sleep(10.0)

    metrics = health.get_health_metrics()["diagnostics"]

    assert metrics["latest"]["snapshot"]["status"] == "ok"
    assert metrics["field_count"] == 1
    assert metrics["latest"]["source"] == "engine"
    assert metrics["freshness"]["is_fresh"] is True

