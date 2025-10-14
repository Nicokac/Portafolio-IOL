"""Tests for the preload worker lazy-import instrumentation."""

from __future__ import annotations

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

    preload._preload_target(("lib_a", "lib_b"))

    assert any("library=lib_a" in event and "duration_ms=" in event for event in events)
    assert any("library=lib_b" in event and "duration_ms=" in event for event in events)
    assert all(event.startswith("preload") for event in events)
