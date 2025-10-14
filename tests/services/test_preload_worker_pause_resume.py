from __future__ import annotations

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
    assert any(entry.startswith("preload") for entry in events)


def test_resume_ignored_when_worker_already_running(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_wait_for_completion_returns_false_when_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class BlockingEvent:
        def wait(self, timeout: float | None = None) -> bool:
            return False

    monkeypatch.setattr(preload, "_FINISHED_EVENT", BlockingEvent(), raising=False)
    assert preload.wait_for_preload_completion(timeout=0.01) is False
