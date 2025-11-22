"""Tests for the preload worker lazy-import instrumentation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import services.preload_worker as preload


def test_preload_worker_logs_duration(monkeypatch):
    events: list[str] = []

    monkeypatch.setattr(preload, "log_startup_event", events.append)

    counter = {"value": 0.0}

    def fake_perf_counter() -> float:
        counter["value"] += 0.05
        return counter["value"]

    def fake_import(name: str) -> SimpleNamespace:
        return SimpleNamespace(module=name)

    monkeypatch.setattr(preload.time, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(preload.importlib, "import_module", fake_import)

    preload._RESUME_EVENT.set()
    preload._FINISHED_EVENT.clear()
    try:
        preload._preload_target(("lib_a", "lib_b"))
    finally:
        preload._reset_events()

    payloads = [json.loads(event) for event in events]
    library_events = [event for event in payloads if event["event"] == "preload_library"]

    assert any(event["module_name"] == "lib_a" for event in library_events)
    assert any(event["module_name"] == "lib_b" for event in library_events)
    assert all("timestamp" in event for event in library_events)
