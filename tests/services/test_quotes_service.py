"""Unit tests validating quote metadata normalization and bulk fetch flows."""

from __future__ import annotations

from typing import Iterable, Tuple

import pytest

from services import cache as cache_module
from services.quote_rate_limit import quote_rate_limiter


QuoteKey = Tuple[str, str]


@pytest.fixture(autouse=True)
def _clear_quotes_cache(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Ensure each test runs with a clean in-memory and persisted cache."""

    cache_module.fetch_quotes_bulk.clear()
    cache_module._QUOTE_CACHE.clear()
    cache_module._QUOTE_PERSIST_CACHE = None
    monkeypatch.setattr(cache_module, "_QUOTE_PERSIST_PATH", tmp_path / "quotes.json")
    yield
    cache_module.fetch_quotes_bulk.clear()
    cache_module._QUOTE_CACHE.clear()
    cache_module._QUOTE_PERSIST_CACHE = None


@pytest.fixture
def mixed_pairs() -> list[QuoteKey]:
    """Canonical list of quote identifiers used across tests."""

    return [("bcba", "GGAL"), ("nyse", "AAPL"), ("bcba", "ALUA")]


class DummyBulkClient:
    """Fake bulk client returning heterogeneous quote payloads."""

    __module__ = "infrastructure.iol.client"

    def __init__(self) -> None:
        self._last_bulk_stats = {}

    def get_quotes_bulk(self, items: Iterable[QuoteKey]):
        return {
            ("bcba", "GGAL"): {
                "last": 100.0,
                "chg_pct": 1.0,
                "provider": "iol",
                "moneda_origen": "ARS",
            },
            ("nyse", "AAPL"): {
                "last": 200.5,
                "chg_pct": 0.75,
                "provider": "polygon",
                "currency": "USD",
                "fx_aplicado": 350.5,
            },
            ("bcba", "ALUA"): {
                "last": None,
                "chg_pct": None,
                "provider": "legacy",
            },
        }


def test_fetch_quotes_bulk_preserves_metadata_from_bulk(mixed_pairs: list[QuoteKey]):
    client = DummyBulkClient()

    result = cache_module.fetch_quotes_bulk(client, mixed_pairs)

    ggal = result[("bcba", "GGAL")]
    assert ggal["moneda_origen"] == "ARS"
    assert ggal["proveedor_original"] == "iol"
    assert ggal["fx_aplicado"] is None

    aapl = result[("nyse", "AAPL")]
    assert aapl["moneda_origen"] == "USD"
    assert aapl["proveedor_original"] == "polygon"
    assert aapl["fx_aplicado"] == pytest.approx(350.5)

    alua = result[("bcba", "ALUA")]
    assert alua["moneda_origen"] is None
    assert alua["proveedor_original"] == "legacy"


class DummySingleClient:
    """Fake client exposing per-symbol fetching only to trigger fallback path."""

    __module__ = "infrastructure.iol.client"
    get_quotes_bulk = None

    def get_quote(self, market: str, symbol: str, panel: str | None = None):
        return {"last": 55.0, "chg_pct": 0.2, "currency": "USD_CCL"}


def test_fetch_quotes_bulk_per_symbol_infers_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = DummySingleClient()

    monkeypatch.setattr(quote_rate_limiter, "wait_for_slot", lambda provider: None)
    monkeypatch.setattr(
        quote_rate_limiter,
        "penalize",
        lambda provider, minimum_wait=None: 0.0,
    )

    result = cache_module.fetch_quotes_bulk(client, [("bcba", "GGDX")])
    quote = result[("bcba", "GGDX")]

    assert quote["provider"] == "iol"
    assert quote["proveedor_original"] == "iol"
    assert quote["moneda_origen"] == "USD_CCL"
    assert quote["fx_aplicado"] is None
