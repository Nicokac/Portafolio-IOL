import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pytest

from services import cache as cache_module

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DummyBulkClient:
    """Fake client exposing ``get_quotes_bulk`` for bulk-mode tests."""

    def __init__(self, stats: Dict[str, Any] | None = None) -> None:
        self._last_bulk_stats = stats or {"rate_limited": 3}

    def get_quotes_bulk(self, items: Iterable[Tuple[str, str]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        return {
            ("bcba", "GGAL"): {"last": 100.0, "chg_pct": 1.0, "provider": "iol"},
            ("bcba", "ALUA"): {"last": 200.0, "chg_pct": -0.5, "provider": "legacy"},
        }


class DummySingleClient:
    """Fake client exposing ``get_quote`` only to trigger per-symbol fallback."""

    def get_quote(self, market: str, symbol: str, panel: str | None = None) -> Dict[str, Any]:
        if symbol == "GGAL":
            return {"last": 120.0, "chg_pct": 0.25, "provider": "iol"}
        return {"last": None, "chg_pct": None, "provider": "error"}


@pytest.fixture(autouse=True)
def _clear_fetch_quotes_cache():
    cache_module.fetch_quotes_bulk.clear()
    cache_module._QUOTE_CACHE.clear()
    cache_module._QUOTE_PERSIST_CACHE = None
    if cache_module._QUOTE_PERSIST_PATH.exists():
        cache_module._QUOTE_PERSIST_PATH.unlink()
    yield
    cache_module.fetch_quotes_bulk.clear()
    cache_module._QUOTE_CACHE.clear()
    cache_module._QUOTE_PERSIST_CACHE = None
    if cache_module._QUOTE_PERSIST_PATH.exists():
        cache_module._QUOTE_PERSIST_PATH.unlink()


def _collect_info_logs(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.message
        for record in caplog.records
        if record.levelno == logging.INFO and record.name == "services.cache" and "quotes processed" in record.message
    ]


def test_fetch_quotes_bulk_logs_single_info_for_bulk_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = DummyBulkClient()
    items = [("bcba", "GGAL"), ("bcba", "ALUA")]

    with caplog.at_level(logging.INFO, logger="services.cache"):
        result = cache_module.fetch_quotes_bulk(client, items)

    assert len(result) == 2
    messages = _collect_info_logs(caplog)
    assert len(messages) == 1
    message = messages[0]
    assert "2 quotes processed" in message
    assert "avg" in message
    assert "qps" in message
    assert "fallbacks=1" in message or "fallbacks=2" in message
    assert "rate_limited=3" in message


def test_fetch_quotes_bulk_logs_single_info_for_per_symbol(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = DummySingleClient()
    items = [("bcba", "GGAL"), ("bcba", "ALUA")]

    # Avoid actual waiting during tests.
    monkeypatch.setattr(cache_module.quote_rate_limiter, "wait_for_slot", lambda provider: None)
    monkeypatch.setattr(
        cache_module.quote_rate_limiter,
        "penalize",
        lambda provider, minimum_wait=None: 0.0,
    )

    with caplog.at_level(logging.INFO, logger="services.cache"):
        result = cache_module.fetch_quotes_bulk(client, items)

    assert len(result) == 2
    messages = _collect_info_logs(caplog)
    assert len(messages) == 1
    message = messages[0]
    assert "2 quotes processed" in message
    assert "avg" in message
    assert "qps" in message
    # One quote succeeds, the other fails.
    assert "fresh=1" in message
    assert "errors=1" in message
