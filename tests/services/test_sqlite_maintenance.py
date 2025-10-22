from __future__ import annotations

import importlib
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pytest

from services.performance_timer import PerformanceEntry


@pytest.mark.slow
def test_sqlite_maintenance_stress(tmp_path, monkeypatch):
    """Stress the maintenance cycle with large caches and ensure cleanup occurs."""

    # --- Prepare MarketDataCache backend ---
    from services.cache import market_data_cache as cache_module

    backend = cache_module._SQLiteBackend(tmp_path / "market_cache.db")
    monkeypatch.setattr(cache_module, "_BACKEND", backend)
    monkeypatch.setattr(cache_module, "_initialise_backend", lambda: backend)

    cache = cache_module.create_persistent_cache("stress")

    base_time = time.time() - 4 * 3600
    monkeypatch.setattr(cache_module.time, "time", lambda: base_time)

    payload = {"buffer": "x" * 2048}
    total_cache_entries = 500
    for idx in range(total_cache_entries):
        cache.set(f"key-{idx}", payload, ttl=1.0)

    now_ts = base_time + 4 * 3600
    monkeypatch.setattr(cache_module.time, "time", lambda: now_ts)

    # --- Prepare performance_store database ---
    monkeypatch.setenv("PERFORMANCE_DB_PATH", str(tmp_path / "perf.db"))
    import services.performance_store as perf_store

    importlib.reload(perf_store)
    monkeypatch.setattr(perf_store, "app_env", "prod")

    old_timestamp = datetime.now(timezone.utc) - timedelta(days=10)
    recent_timestamp = datetime.now(timezone.utc)

    old_entries = 300
    for idx in range(old_entries):
        perf_store.store_entry(
            PerformanceEntry(
                timestamp=old_timestamp.isoformat(),
                label=f"batch-old-{idx}",
                duration_s=0.5,
                cpu_percent=None,
                ram_percent=None,
                extras={"batch": "old"},
                module="tests",
                success=True,
            )
        )

    perf_store.store_entry(
        PerformanceEntry(
            timestamp=recent_timestamp.isoformat(),
            label="batch-new",
            duration_s=1.0,
            cpu_percent=None,
            ram_percent=None,
            extras={"batch": "new"},
            module="tests",
            success=True,
        )
    )

    import services.maintenance.sqlite_maintenance as maintenance

    importlib.reload(maintenance)

    reports = maintenance.run_sqlite_maintenance_now(
        reason="pytest",
        now=now_ts,
        vacuum=True,
    )

    report_map = {report["database"]: report for report in reports}

    market_report = report_map.get("market_data_cache")
    assert market_report is not None
    assert market_report["deleted"] >= total_cache_entries
    assert market_report["size_after"] <= market_report["size_before"]

    perf_report = report_map.get("performance_store")
    assert perf_report is not None
    assert perf_report["deleted"] >= old_entries
    assert perf_report["size_after"] <= perf_report["size_before"]

    with sqlite3.connect(perf_store.get_database_path()) as conn:
        remaining = conn.execute("SELECT COUNT(*) FROM performance_metrics").fetchone()[0]
    assert remaining == 1
