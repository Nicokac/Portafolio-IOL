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


def test_adaptive_cache_lock_warns_on_long_wait(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    calls = deque([0.0, 6.5, 6.5])

    def _fake_monotonic() -> float:
        return calls.popleft()

    monkeypatch.setattr(adaptive_module.time, "monotonic", _fake_monotonic)

    with caplog.at_level(logging.WARNING):
        with adaptive_module.adaptive_cache_lock:
            pass

    assert any("demoró" in record.message for record in caplog.records)


def test_adaptive_cache_lock_warns_on_long_hold(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    calls = deque([0.0, 0.0, 6.5])

    def _fake_monotonic() -> float:
        return calls.popleft()

    monkeypatch.setattr(adaptive_module.time, "monotonic", _fake_monotonic)

    with caplog.at_level(logging.WARNING):
        with adaptive_module.adaptive_cache_lock:
            pass

    assert any("retenido" in record.message for record in caplog.records)
