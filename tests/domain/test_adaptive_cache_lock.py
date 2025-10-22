import logging
import threading
import time
from collections import deque

import pytest

import domain.adaptive_cache_lock as adaptive_module


def test_adaptive_cache_lock_is_mutually_exclusive() -> None:
    order: list[str] = []
    release_event = threading.Event()

    def _worker() -> None:
        with adaptive_module.adaptive_cache_lock:
            order.append("worker")
            release_event.set()

    with adaptive_module.adaptive_cache_lock:
        order.append("main")
        worker = threading.Thread(target=_worker)
        worker.start()
        time.sleep(0.1)
        # Worker should still be waiting for the lock.
        assert order == ["main"]

    release_event.wait(timeout=2.0)
    worker.join(timeout=2.0)

    assert order == ["main", "worker"]


def test_adaptive_cache_lock_is_reentrant() -> None:
    with adaptive_module.adaptive_cache_lock:
        assert adaptive_module.adaptive_cache_lock.locked()
        with adaptive_module.adaptive_cache_lock:
            assert adaptive_module.adaptive_cache_lock.locked()
        assert adaptive_module.adaptive_cache_lock.locked()
    assert not adaptive_module.adaptive_cache_lock.locked()


def test_adaptive_cache_lock_warns_on_long_wait(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = deque([0.0, 5.5, 5.5])

    def _fake_monotonic() -> float:
        return calls.popleft()

    monkeypatch.setattr(adaptive_module.adaptive_cache_lock, "_warn_after", 1.0)
    monkeypatch.setattr(adaptive_module.time, "monotonic", _fake_monotonic)

    with caplog.at_level(logging.WARNING):
        with adaptive_module.adaptive_cache_lock:
            pass

    assert any("demoró" in record.message and "módulo" in record.message for record in caplog.records)


def test_adaptive_cache_lock_warns_on_long_hold(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = deque([0.0, 0.0, 5.5])

    def _fake_monotonic() -> float:
        return calls.popleft()

    monkeypatch.setattr(adaptive_module.adaptive_cache_lock, "_warn_after", 1.0)
    monkeypatch.setattr(adaptive_module.time, "monotonic", _fake_monotonic)

    with caplog.at_level(logging.WARNING):
        with adaptive_module.adaptive_cache_lock:
            pass

    assert any("retenido" in record.message and "owner" in record.message for record in caplog.records)


def test_adaptive_cache_lock_warns_on_prolonged_hold(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = deque([0.0, 0.0, 130.0])

    def _fake_monotonic() -> float:
        return calls.popleft()

    monkeypatch.setattr(adaptive_module.adaptive_cache_lock, "_warn_after", 1.0)
    monkeypatch.setattr(adaptive_module.time, "monotonic", _fake_monotonic)

    with caplog.at_level(logging.WARNING):
        with adaptive_module.adaptive_cache_lock:
            pass

    assert any("Retención prolongada" in record.message for record in caplog.records)


def test_adaptive_cache_lock_releases_on_exception() -> None:
    with pytest.raises(RuntimeError):
        with adaptive_module.adaptive_cache_lock:
            raise RuntimeError("boom")

    assert not adaptive_module.adaptive_cache_lock.locked()


def test_run_in_background_returns_without_blocking() -> None:
    event = threading.Event()

    def _worker() -> None:
        time.sleep(0.2)
        event.set()

    start = time.perf_counter()
    thread = adaptive_module.run_in_background(_worker, name="test-bg", daemon=True)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1
    assert thread.is_alive()

    assert event.wait(timeout=2.0)
    thread.join(timeout=1.0)
