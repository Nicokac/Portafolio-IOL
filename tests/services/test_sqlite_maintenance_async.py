from __future__ import annotations

import csv
import importlib
import time
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_async_state(monkeypatch):
    import services.maintenance.sqlite_maintenance as maintenance

    importlib.reload(maintenance)
    maintenance._reset_async_cache_state_for_tests()
    yield
    maintenance._reset_async_cache_state_for_tests()


def _prepare_sqlite_cache(tmp_path, monkeypatch):
    from services.cache import market_data_cache as cache_module

    importlib.reload(cache_module)

    backend = cache_module._SQLiteBackend(tmp_path / "market_cache.db")
    monkeypatch.setattr(cache_module, "_BACKEND", backend)
    monkeypatch.setattr(cache_module, "_initialise_backend", lambda: backend)
    return cache_module, backend


def test_async_sqlite_maintenance_generates_telemetry(tmp_path, monkeypatch):
    cache_module, backend = _prepare_sqlite_cache(tmp_path, monkeypatch)
    import services.maintenance.sqlite_maintenance as maintenance

    metrics_path = Path("performance_metrics_13.csv")
    if metrics_path.exists():
        metrics_path.unlink()

    log_path = Path("services/maintenance/logs/sqlite_maintenance.log")
    if log_path.exists():
        log_path.unlink()

    now = time.time()
    payload = {"payload": "x" * 65536}
    for idx in range(200):
        backend.set(f"expired-{idx}", payload, now - 3600)
    for idx in range(5):
        backend.set(f"active-{idx}", payload, now + 3600)

    cache_path = backend.database_path()
    size_before = cache_path.stat().st_size

    monkeypatch.setattr(maintenance, "_dynamic_threshold", lambda size_mb: size_mb * 0.5)

    original_run = cache_module.run_persistent_cache_maintenance

    def delayed_run(**kwargs):
        time.sleep(0.25)
        return original_run(**kwargs)

    monkeypatch.setattr(cache_module, "run_persistent_cache_maintenance", delayed_run)

    start = time.perf_counter()
    scheduled = maintenance.run_sqlite_maintenance(force=True)
    elapsed = time.perf_counter() - start

    assert scheduled is True
    assert elapsed < 0.2
    assert maintenance.is_sqlite_maintenance_running()

    result = maintenance.wait_for_sqlite_maintenance(timeout=5.0)
    assert result is not None
    assert not maintenance.is_sqlite_maintenance_running()

    size_after = cache_path.stat().st_size
    assert size_after < size_before

    before_mb = float(result["db_size_before_mb"])
    after_mb = float(result["db_size_after_mb"])
    freed_mb = float(result["freed_space_mb"])

    assert before_mb >= after_mb
    assert freed_mb > 0
    assert result["was_async"] is True
    assert int(result["tables_cleaned"]) >= 1

    assert metrics_path.exists()
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "expected telemetry rows to be recorded"
    latest = rows[-1]
    assert float(latest["db_size_before_mb"]) >= float(latest["db_size_after_mb"])
    assert float(latest["freed_space_mb"]) >= freed_mb
    assert latest["was_async"] in {"True", "true", "1", "True"}

    cooldown_skip = maintenance.run_sqlite_maintenance()
    assert cooldown_skip is False
