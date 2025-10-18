import time
from types import SimpleNamespace

from shared import user_actions


def test_logging_latency(monkeypatch, tmp_path):
    log_path = tmp_path / "user_actions.csv"
    monkeypatch.setattr(user_actions, "_LOG_PATH", log_path)
    stub_streamlit = SimpleNamespace(session_state={})
    monkeypatch.setattr(user_actions, "st", stub_streamlit, raising=False)
    monkeypatch.setattr(user_actions, "_resolve_user_id", lambda: "test-user")
    user_actions._reset_for_tests()
    user_actions.log_user_action("warmup", {"index": -1})
    assert user_actions.wait_for_flush(2.0)

    start = time.perf_counter()
    for idx in range(1000):
        user_actions.log_user_action(
            "bulk_event",
            {"index": idx},
            dataset_hash="perf-hash",
        )
    elapsed = time.perf_counter() - start
    assert elapsed / 1000.0 <= 0.005
    assert user_actions.wait_for_flush(5.0)

    with log_path.open(newline="", encoding="utf-8") as handle:
        lines = handle.readlines()
    # 1000 events + header
    assert len(lines) >= 1001
