import csv
import time

import csv
import time
from types import SimpleNamespace

import pytest

from shared import user_actions


@pytest.fixture(autouse=True)
def reset_user_action_logger(monkeypatch, tmp_path):
    log_path = tmp_path / "user_actions.csv"
    monkeypatch.setattr(user_actions, "_LOG_PATH", log_path)
    stub_streamlit = SimpleNamespace(session_state={})
    monkeypatch.setattr(user_actions, "st", stub_streamlit, raising=False)
    monkeypatch.setattr(user_actions, "_resolve_user_id", lambda: "test-user")
    user_actions._reset_for_tests()
    yield
    user_actions._reset_for_tests()


def test_log_user_action_is_non_blocking():
    user_actions.log_user_action("warmup", {"index": -1})
    assert user_actions.wait_for_flush(2.0)
    start = time.perf_counter()
    for idx in range(50):
        user_actions.log_user_action("test_click", {"index": idx})
    elapsed = time.perf_counter() - start
    assert elapsed < 0.2, "logging should not block the caller"
    assert user_actions.wait_for_flush(2.0)


def test_user_actions_written_to_csv(tmp_path):
    user_actions.log_user_action(
        "load_portfolio_table",
        {"tab": "portafolio"},
        dataset_hash="hash-1",
        latency_ms=12.5,
    )
    user_actions.log_user_action(
        "tab_change",
        {"tab": "resumen"},
        dataset_hash="hash-1",
    )
    assert user_actions.wait_for_flush(2.0)
    with user_actions._LOG_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 2
    first = rows[0]
    assert first["action"] == "load_portfolio_table"
    assert first["dataset_hash"] == "hash-1"
    assert first["latency_ms"] == "12.50"


def test_worker_failure_reports_telemetry(monkeypatch):
    telemetry_calls = []

    def fake_log_default_telemetry(**kwargs):
        telemetry_calls.append(kwargs)

    monkeypatch.setattr(user_actions, "_write_rows", lambda events: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(user_actions, "log_default_telemetry", fake_log_default_telemetry, raising=False)

    user_actions.log_user_action("test_event", "detail")
    assert user_actions.wait_for_flush(2.0)
    assert telemetry_calls, "worker errors should be reported via telemetry"
    assert telemetry_calls[0]["phase"] == "user_action_logger_error"

    # Restore normal behaviour and ensure the logger recovers.
    monkeypatch.setattr(user_actions, "_write_rows", lambda events: None)
    user_actions.log_user_action("recovery_event", "detail")
    assert user_actions.wait_for_flush(2.0)
