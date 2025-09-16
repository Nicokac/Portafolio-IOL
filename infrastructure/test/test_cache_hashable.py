import sys
import time
from threading import Thread

from shared.cache import cache

cache_module = sys.modules["shared.cache"]


def test_cache_data_handles_list_arguments():
    calls = {"n": 0}

    @cache.cache_data()
    def sum_list(values):
        calls["n"] += 1
        return sum(values)

    assert sum_list([1, 2]) == 3
    assert sum_list([1, 2]) == 3
    assert calls["n"] == 1

    sum_list.clear()


def test_cache_thread_safety():
    calls = {"n": 0}

    @cache.cache_data()
    def identity(x):
        calls["n"] += 1
        return x

    def worker():
        for _ in range(3):
            assert identity(1) == 1

    threads = [Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls["n"] == 1
    identity.clear()


def test_cache_expiration_cleanup():
    calls = {"n": 0}

    @cache.cache_data(ttl=1)
    def identity(x):
        calls["n"] += 1
        return x

    # First call caches value 1
    assert identity(1) == 1
    assert calls["n"] == 1
    # Wait for entry to expire
    time.sleep(1.1)
    # Access with new key to trigger cleanup
    assert identity(2) == 2
    assert calls["n"] == 2

    # Internal cache should only hold the second key after cleanup
    cache_dicts = [cell.cell_contents for cell in identity.__closure__ if isinstance(cell.cell_contents, dict)]
    assert all(len(d) == 1 for d in cache_dicts)
    key = next(iter(cache_dicts[0].keys()))
    assert key[0][0] == 2

    identity.clear()


def test_cache_data_maxsize_evicts_old_entries():
    calls = {"n": 0}

    @cache.cache_data(maxsize=2)
    def identity(x):
        calls["n"] += 1
        return x

    assert identity(1) == 1
    assert identity(2) == 2
    assert calls["n"] == 2

    assert identity(2) == 2
    assert calls["n"] == 2

    assert identity(3) == 3
    assert calls["n"] == 3

    assert identity(1) == 1
    assert calls["n"] == 4

    identity.clear()


def test_cache_resource_maxsize_eviction(monkeypatch):
    fake_session = {"session_id": "test"}
    monkeypatch.setattr(cache_module.st, "session_state", fake_session)

    calls = {"n": 0}

    @cache.cache_resource(maxsize=2)
    def build(name):
        calls["n"] += 1
        return {"name": name}

    first = build("a")
    second = build("b")

    assert calls["n"] == 2
    assert build("a") is first
    assert calls["n"] == 2

    third = build("c")
    assert calls["n"] == 3
    assert third == {"name": "c"}

    assert build("b") is second
    assert calls["n"] == 3

    new_first = build("a")
    assert calls["n"] == 4
    assert new_first == {"name": "a"}
    assert new_first is not first

    build.clear()
