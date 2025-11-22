from __future__ import annotations

import importlib
import sys
import time
import json
import types

import pytest

import services.preload_worker as preload


@pytest.fixture(autouse=True)
def reset_worker_state():
    preload.reset_worker_for_tests()
    yield
    try:
        preload.wait_for_preload_completion(timeout=0.1)
    except Exception:
        pass
    preload.reset_worker_for_tests()


def test_preload_worker_pauses_and_resumes(monkeypatch: pytest.MonkeyPatch) -> None:
    imported: list[str] = []

    def fake_import(name: str) -> None:
        imported.append(name)

    monkeypatch.setattr(
        preload,
        "importlib",
        types.SimpleNamespace(import_module=fake_import),
        raising=False,
    )
    monkeypatch.setattr(
        preload,
        "_get_metric_updaters",
        lambda: (lambda *_args, **_kwargs: None, lambda *_a, **_k: None),
    )

    events: list[str] = []
    monkeypatch.setattr(preload, "log_startup_event", events.append)

    started = preload.start_preload_worker(libraries=("lib_a", "lib_b"), paused=True)
    assert started is True
    assert imported == []  # still paused

    resumed = preload.resume_preload_worker(delay_seconds=0.0)
    assert resumed is True
    assert preload.wait_for_preload_completion(timeout=1.0) is True
    assert imported == ["lib_a", "lib_b"]

    metrics = preload.get_preload_metrics()
    assert metrics.status == preload.PreloadPhase.COMPLETED
    assert metrics.total_ms is not None and metrics.total_ms >= 0.0
    assert metrics.durations_ms["lib_a"] is not None
    payloads = [json.loads(entry) for entry in events]
    assert any(event["event"] == "preload_total" for event in payloads)


def test_resume_ignored_when_worker_already_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preload,
        "importlib",
        types.SimpleNamespace(import_module=lambda name: None),
        raising=False,
    )
    monkeypatch.setattr(
        preload,
        "_get_metric_updaters",
        lambda: (lambda *_args, **_kwargs: None, lambda *_a, **_k: None),
    )

    preload.start_preload_worker(libraries=("lib_x",), paused=True)
    first = preload.resume_preload_worker(delay_seconds=0.0)
    assert first is True
    second = preload.resume_preload_worker(delay_seconds=0.0)
    assert second is False
    assert preload.wait_for_preload_completion(timeout=1.0) is True


def test_wait_for_completion_returns_false_when_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingEvent:
        def wait(self, timeout: float | None = None) -> bool:
            return False

        def clear(self) -> None:
            return None

    monkeypatch.setattr(preload, "_FINISHED_EVENT", BlockingEvent(), raising=False)
    assert preload.wait_for_preload_completion(timeout=0.01) is False


def test_preload_worker_import_is_lightweight(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "services.preload_worker"
    sys.modules.pop(module_name, None)

    heavy_libs = ("pandas", "plotly", "statsmodels")
    saved_heavy = {name: sys.modules.get(name) for name in heavy_libs}
    for name in heavy_libs:
        sys.modules.pop(name, None)

    original_import_module = importlib.import_module
    imported: list[str] = []

    def tracking_import(name: str, package: str | None = None):
        imported.append(name)
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", tracking_import)

    module = None
    start = time.perf_counter()
    try:
        module = importlib.import_module(module_name)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        assert elapsed_ms <= 50.0
        for name in heavy_libs:
            assert name not in imported
            assert name not in sys.modules
    finally:
        if module is not None:
            module.reset_worker_for_tests()
        for name, original in saved_heavy.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
