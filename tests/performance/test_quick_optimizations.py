"""Regression tests for quick performance optimizations."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import pytest

from services.cache import quotes as quotes_module
from shared.telemetry import DEFAULT_TELEMETRY_FILES, log_default_telemetry


@pytest.fixture(autouse=True)
def _clean_metrics_files() -> None:
    """Ensure telemetry CSVs start empty for each test."""

    for path in DEFAULT_TELEMETRY_FILES:
        metrics_path = Path(path)
        if metrics_path.exists():
            metrics_path.unlink()


def test_telemetry_appends_rows_to_metrics_files() -> None:
    """Telemetry utility should append structured rows to both CSV logs."""

    log_default_telemetry(
        phase="quotes_refresh",
        elapsed_s=0.25,
        dataset_hash="hash123",
        subbatch_avg_s=0.05,
        ui_total_load_ms=1234.0,
    )

    for path in DEFAULT_TELEMETRY_FILES:
        metrics_path = Path(path)
        assert metrics_path.exists()
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        row = rows[0]
        assert row["phase"] == "quotes_refresh"
        assert row["dataset_hash"] == "hash123"
        assert row["ui_total_load_ms"] == "1234.00"
        assert row["subbatch_avg_s"] == "0.050000"


def test_warm_start_reduces_first_fetch(monkeypatch, tmp_path) -> None:
    """Warm-start should serve cached quotes faster than a cold refresh."""

    # Reset module state
    quotes_module.fetch_quotes_bulk.clear()
    quotes_module._QUOTE_CACHE.clear()
    quotes_module._QUOTE_PERSIST_CACHE = None
    quotes_module._QUOTE_WARM_START_APPLIED = False

    persist_path = tmp_path / "quotes_cache.json"
    monkeypatch.setattr(quotes_module, "_QUOTE_PERSIST_PATH", persist_path)

    persisted = {
        "bcba|GGAL|": {
            "data": {"last": 100.0, "provider": "cache", "stale": False},
            "ts": time.time() - 2,
        }
    }
    persist_path.write_text(json.dumps(persisted), encoding="utf-8")

    monkeypatch.setattr(quotes_module, "cache_ttl_quotes", 30)
    monkeypatch.setattr(quotes_module, "QUOTE_STALE_TTL_SECONDS", 30.0)

    class SlowClient:
        def get_quote(self, market: str, symbol: str, panel: str | None = None):
            time.sleep(0.05)
            return {"last": 101.0, "provider": "iol", "asof": "now"}

    client = SlowClient()
    items = [("bcba", "GGAL")]

    start = time.perf_counter()
    warm_payload = quotes_module.fetch_quotes_bulk(client, items)
    warm_elapsed = time.perf_counter() - start

    assert warm_payload.get(("bcba", "GGAL"), {}).get("last") == 100.0

    quotes_module.fetch_quotes_bulk.clear()
    quotes_module._QUOTE_CACHE.clear()

    start = time.perf_counter()
    cold_payload = quotes_module.fetch_quotes_bulk(client, items)
    cold_elapsed = time.perf_counter() - start

    assert cold_payload.get(("bcba", "GGAL"), {}).get("last") == 101.0

    assert warm_elapsed < cold_elapsed
    assert (cold_elapsed - warm_elapsed) > 0.02
