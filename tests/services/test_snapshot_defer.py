from __future__ import annotations

import threading
import time
from typing import Any, Mapping

import pytest

from services import snapshot_defer
from shared import snapshot as snapshot_async


@pytest.fixture(autouse=True)
def _reset_snapshot_defer():
    snapshot_defer.reset_for_tests()
    yield
    snapshot_defer.reset_for_tests()


def _dummy_persist(
    kind: str, payload: Mapping[str, Any], metadata: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    return {"kind": kind, "payload": payload, "metadata": metadata}


def test_snapshot_persistence_waits_for_ui_idle(monkeypatch):
    call_event = threading.Event()
    calls: list[dict[str, Any]] = []

    def fake_persist_async(**kwargs: Any) -> None:
        calls.append(kwargs)
        call_event.set()

    monkeypatch.setattr(snapshot_async, "persist_async", fake_persist_async)

    snapshot_defer.mark_ui_busy()
    snapshot_defer.queue_snapshot_persistence(
        kind="portfolio",
        payload={"value": 1},
        metadata={"filters": []},
        persist_fn=_dummy_persist,
        dataset_hash="hash-a",
    )

    assert not call_event.wait(0.05)

    snapshot_defer.mark_ui_idle(timestamp=time.perf_counter())
    assert call_event.wait(1.0)
    assert len(calls) == 1


def test_snapshot_persistence_deduplicates_dataset_hash(monkeypatch):
    call_event = threading.Event()
    calls: list[dict[str, Any]] = []

    def fake_persist_async(**kwargs: Any) -> None:
        calls.append(kwargs)
        call_event.set()

    monkeypatch.setattr(snapshot_async, "persist_async", fake_persist_async)

    snapshot_defer.mark_ui_busy()
    snapshot_defer.queue_snapshot_persistence(
        kind="portfolio",
        payload={"value": 1},
        metadata=None,
        persist_fn=_dummy_persist,
        dataset_hash="hash-b",
    )
    snapshot_defer.queue_snapshot_persistence(
        kind="portfolio",
        payload={"value": 2},
        metadata=None,
        persist_fn=_dummy_persist,
        dataset_hash="hash-b",
    )

    snapshot_defer.mark_ui_idle(timestamp=time.perf_counter())
    assert call_event.wait(1.0)
    assert len(calls) == 1


def test_snapshot_defer_enriches_telemetry(monkeypatch):
    call_event = threading.Event()
    telemetry_payloads: list[Mapping[str, object]] = []

    def fake_persist_async(**kwargs: Any) -> None:
        telemetry = kwargs.get("telemetry_fn")
        if callable(telemetry):
            telemetry("snapshot.persist_async", 0.25, "hash-c", {"base": "value"})
        call_event.set()

    monkeypatch.setattr(snapshot_async, "persist_async", fake_persist_async)

    def base_telemetry(
        phase: str,
        elapsed_s: float | None,
        dataset_hash: str | None,
        extra: Mapping[str, object] | None,
    ) -> None:
        telemetry_payloads.append(dict(extra or {}))

    snapshot_defer.mark_ui_busy()
    snapshot_defer.queue_snapshot_persistence(
        kind="portfolio",
        payload={"value": 3},
        metadata=None,
        persist_fn=_dummy_persist,
        dataset_hash="hash-c",
        telemetry_fn=base_telemetry,
    )

    snapshot_defer.mark_ui_idle(timestamp=time.perf_counter() + 0.2)
    assert call_event.wait(1.0)
    assert telemetry_payloads, "Expected telemetry to be emitted"
    enriched = telemetry_payloads[0]
    assert enriched.get("base") == "value"
    assert "snapshot_deferred_count" in enriched
    assert "ui_idle_latency_ms" in enriched
    assert enriched["snapshot_deferred_count"] >= 1
    assert enriched["ui_idle_latency_ms"] >= 0.0
