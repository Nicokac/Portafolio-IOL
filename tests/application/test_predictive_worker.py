from __future__ import annotations

import threading
import time

import pandas as pd
import pytest

from application import predictive_jobs


@pytest.fixture(autouse=True)
def _reset_jobs() -> None:
    predictive_jobs.reset()
    yield
    predictive_jobs.reset()


def test_submit_runs_in_background_and_exposes_latest_result() -> None:
    def _job() -> pd.DataFrame:
        time.sleep(0.1)
        return pd.DataFrame({"value": [1, 2, 3]})

    job_id = predictive_jobs.submit("predict", _job, ttl_seconds=0.3)
    latest = predictive_jobs.get_latest("predict")
    assert latest is not None
    value, metadata = latest
    assert value is None
    assert metadata["job_id"] == job_id
    assert metadata["status"] in {"running", "pending"}

    for _ in range(50):
        snapshot = predictive_jobs.status(job_id)
        if snapshot.get("status") == "finished":
            break
        time.sleep(0.05)
    else:  # pragma: no cover - defensive
        pytest.fail("job did not finish on time")

    latest = predictive_jobs.get_latest("predict")
    assert latest is not None
    value, metadata = latest
    assert isinstance(value, pd.DataFrame)
    assert list(value.columns) == ["value"]
    assert metadata["result_ready"] is True

    time.sleep(0.4)
    latest = predictive_jobs.get_latest("predict")
    assert latest is not None
    value, metadata = latest
    assert value is None
    assert metadata["status"] == "finished"


def test_submit_deduplicates_pending_jobs() -> None:
    called = 0
    unblock = threading.Event()

    def _job() -> str:
        nonlocal called
        called += 1
        unblock.wait(0.1)
        return "ready"

    first = predictive_jobs.submit("dedupe", _job, ttl_seconds=1.0)
    second = predictive_jobs.submit("dedupe", _job, ttl_seconds=1.0)
    assert first == second

    unblock.set()
    for _ in range(50):
        snapshot = predictive_jobs.status(first)
        if snapshot.get("status") == "finished":
            break
        time.sleep(0.02)
    else:  # pragma: no cover - defensive
        pytest.fail("deduped job did not finish")

    assert called == 1
    latest = predictive_jobs.get_latest("dedupe")
    assert latest is not None
    value, metadata = latest
    assert value == "ready"
    assert metadata["job_id"] == first
