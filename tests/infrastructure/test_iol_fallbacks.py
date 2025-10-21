import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import requests

from tests.fixtures.auth import FakeAuth

# Ensure the project root is importable regardless of pytest's invocation path.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infrastructure.iol import client as iol_client_module
from infrastructure.iol.legacy import iol_client as legacy_module
from services import cache as cache_module
from services import ohlc_adapter as ohlc_module


ORIGINAL_PERSIST_QUOTE = cache_module._persist_quote


@pytest.fixture(autouse=True)
def _reset_cache_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure quote caches and metrics hooks do not leak across tests."""

    cache_module._QUOTE_CACHE.clear()
    cache_module._QUOTE_PERSIST_CACHE = None
    monkeypatch.setattr(cache_module, "_persist_quote", lambda *_, **__: None)
    monkeypatch.setattr(cache_module, "record_quote_provider_usage", lambda *_, **__: None)
    monkeypatch.setattr(cache_module, "record_quote_load", lambda *_, **__: None)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> iol_client_module.IOLClient:
    """Return an IOLClient instance with market auth disabled."""

    monkeypatch.setattr(iol_client_module.IOLClient, "_ensure_market_auth", lambda self: None)
    monkeypatch.setattr(iol_client_module, "st", SimpleNamespace(session_state={}))
    return iol_client_module.IOLClient(
        "user", "", auth=FakeAuth(access="access", refresh="refresh")
    )


def _http_error(status: int) -> requests.HTTPError:
    response = SimpleNamespace(status_code=status)
    return requests.HTTPError(response=response)


def test_primary_endpoint_provides_fresh_quote(
    monkeypatch: pytest.MonkeyPatch, client: iol_client_module.IOLClient
) -> None:
    """The new /Titulos/Cotizacion endpoint should yield a fresh quote without fallbacks."""

    recorded: list[tuple[str, dict[str, Any]]] = []

    def record(provider: str, **kwargs: Any) -> None:
        recorded.append((provider, kwargs))

    monkeypatch.setattr(cache_module, "record_quote_provider_usage", record)
    monkeypatch.setattr(cache_module, "_load_persisted_entry", lambda key: None)

    def successful_request(
        self: iol_client_module.IOLClient, method: str, url: str, **kwargs: Any
    ) -> SimpleNamespace:
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {
                "ultimoPrecio": 245.75,
                "variacion": 2.1,
                "moneda": "ARS",
                "fecha": "2024-05-20T15:30:00",
            },
        )

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", successful_request)

    payload = cache_module._get_quote_cached(client, "bcba", "GGAL", ttl=60)

    assert payload["provider"] == "iol"
    assert payload["last"] == pytest.approx(245.75)
    assert payload.get("stale") is not True
    assert any(entry[0] == "iol" and entry[1].get("stale") is False for entry in recorded)


def test_primary_endpoint_missing_last_triggers_legacy(
    monkeypatch: pytest.MonkeyPatch, client: iol_client_module.IOLClient
) -> None:
    """A missing last price from v2 should trigger the legacy fallback path."""

    recorded: list[tuple[str, dict[str, Any]]] = []

    def record(provider: str, **kwargs: Any) -> None:
        recorded.append((provider, kwargs))

    legacy_calls: list[dict[str, Any]] = []

    def missing_last_request(
        self: iol_client_module.IOLClient, method: str, url: str, **kwargs: Any
    ) -> SimpleNamespace:
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {
                "variacion": 1.5,
                "moneda": "ARS",
                "fecha": "2024-05-20T15:30:00",
            },
        )

    class LegacyStub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def get_quote(self, **kwargs: Any) -> dict[str, Any]:
            legacy_calls.append(kwargs)
            return {
                "last": 101.25,
                "chg_pct": 1.5,
                "asof": "2024-05-20T15:35:00",
                "provider": "legacy",
            }

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", missing_last_request)
    monkeypatch.setattr("infrastructure.iol.legacy.iol_client.IOLClient", LegacyStub)
    monkeypatch.setattr(iol_client_module, "record_quote_provider_usage", record)
    monkeypatch.setattr(cache_module, "record_quote_provider_usage", record)
    monkeypatch.setattr(cache_module, "_load_persisted_entry", lambda key: None)

    payload = cache_module._get_quote_cached(client, "bcba", "GGAL", ttl=60)

    assert legacy_calls, "Legacy client should be invoked when last price is missing"
    assert payload["provider"] == "legacy"
    assert payload["last"] == pytest.approx(101.25)
    assert any(entry[0] == "iol" and entry[1].get("source") == "v2-missing-last" for entry in recorded)


def test_get_quote_uses_legacy_on_primary_http_500(
    monkeypatch: pytest.MonkeyPatch, client: iol_client_module.IOLClient
) -> None:
    """When the primary API returns 500, the legacy client should provide the quote."""

    def failing_request(self, method: str, url: str, **kwargs):  # type: ignore[no-untyped-def]
        raise _http_error(500)

    class LegacyStub:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_quote(self, **kwargs):  # type: ignore[no-untyped-def]
            return {
                "last": 123.45,
                "chg_pct": 1.5,
                "asof": "2024-01-01T10:00:00",
                "provider": "legacy",
            }

    class AdapterStub:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def _make_cache_key(self, symbol, params):  # type: ignore[no-untyped-def]
            return "stub"

        def fetch(self, symbol, **params):  # type: ignore[no-untyped-def]
            raise AssertionError("OHLC adapter should not be used when legacy succeeds")

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", failing_request)
    monkeypatch.setattr("infrastructure.iol.legacy.iol_client.IOLClient", LegacyStub)
    monkeypatch.setattr("services.ohlc_adapter.OHLCAdapter", AdapterStub)
    monkeypatch.setattr(cache_module, "_load_persisted_entry", lambda key: None)

    payload = cache_module._get_quote_cached(client, "bcba", "GGAL", ttl=60)

    assert payload["provider"] == "legacy"


def test_get_quote_uses_ohlc_adapter_after_legacy_failure(
    monkeypatch: pytest.MonkeyPatch, client: iol_client_module.IOLClient
) -> None:
    """When both the primary and legacy clients fail, the OHLC adapter should provide the data."""

    def failing_request(self, method: str, url: str, **kwargs):  # type: ignore[no-untyped-def]
        raise _http_error(500)

    class LegacyStub:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_quote(self, **kwargs):  # type: ignore[no-untyped-def]
            raise _http_error(429)

    class AdapterStub:
        def __init__(self, *args, **kwargs) -> None:
            self._cache: dict[str, SimpleNamespace] = {}

        def _make_cache_key(self, symbol, params):  # type: ignore[no-untyped-def]
            return f"{symbol}|{params.get('period')}|{params.get('interval')}"

        def fetch(self, symbol, **params):  # type: ignore[no-untyped-def]
            key = self._make_cache_key(symbol, params)
            self._cache[key] = SimpleNamespace(provider="alpha_vantage")
            index = pd.to_datetime(["2024-01-01", "2024-01-02"])
            frame = pd.DataFrame({"Close": [100.0, 101.0]}, index=index)
            return frame

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", failing_request)
    monkeypatch.setattr("infrastructure.iol.legacy.iol_client.IOLClient", LegacyStub)
    monkeypatch.setattr("services.ohlc_adapter.OHLCAdapter", AdapterStub)
    monkeypatch.setattr(cache_module, "_load_persisted_entry", lambda key: None)

    payload = cache_module._get_quote_cached(client, "bcba", "YPFD", ttl=60)

    assert payload["provider"] == "av"
    assert payload["last"] == pytest.approx(101.0)


def test_get_quote_returns_stale_from_persistent_cache_when_all_fallbacks_fail(
    monkeypatch: pytest.MonkeyPatch, client: iol_client_module.IOLClient
) -> None:
    """If every live provider fails, a stale entry from the persistent cache should be used."""

    def failing_request(self, method: str, url: str, **kwargs):  # type: ignore[no-untyped-def]
        raise _http_error(500)

    class LegacyStub:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_quote(self, **kwargs):  # type: ignore[no-untyped-def]
            raise _http_error(429)

    class AdapterStub:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def _make_cache_key(self, symbol, params):  # type: ignore[no-untyped-def]
            return "stub"

        def fetch(self, symbol, **params):  # type: ignore[no-untyped-def]
            raise RuntimeError("adapter failure")

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", failing_request)
    monkeypatch.setattr("infrastructure.iol.legacy.iol_client.IOLClient", LegacyStub)
    monkeypatch.setattr("services.ohlc_adapter.OHLCAdapter", AdapterStub)

    stale_entry = (
        {"last": 98.0, "chg_pct": -1.0, "provider": None, "stale": True},
        time.time(),
    )
    monkeypatch.setattr(cache_module, "_load_persisted_entry", lambda key: stale_entry)

    payload = cache_module._get_quote_cached(client, "bcba", "CEPU", ttl=60)

    assert payload["provider"] == "stale"
    assert payload["provider"] is not None
    assert payload["stale"] is True


def test_fetch_quotes_bulk_uses_persisted_price_when_bulk_returns_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = SimpleNamespace(session_state={})
    monkeypatch.setattr(cache_module, "st", fake_st)
    monkeypatch.setattr("shared.cache.st", fake_st, raising=False)

    recorded: list[tuple[str, dict[str, Any]]] = []

    def record(provider: str, **kwargs: Any) -> None:
        recorded.append((provider, kwargs))

    stale_payload = {
        "last": 123.45,
        "chg_pct": 1.2,
        "provider": "iol",
        "asof": "2024-01-01T00:00:00",
    }
    ts = time.time()
    monkeypatch.setattr(cache_module, "_load_persisted_entry", lambda key: (stale_payload, ts))
    monkeypatch.setattr(cache_module, "record_quote_provider_usage", record)
    monkeypatch.setattr(cache_module, "QUOTE_STALE_TTL_SECONDS", 60.0)

    class DummyClient:
        def get_quotes_bulk(self, items):  # type: ignore[no-untyped-def]
            return {tuple(items[0]): {"last": None, "provider": "iol"}}

    cache_module.fetch_quotes_bulk.clear()

    result = cache_module.fetch_quotes_bulk(DummyClient(), [("bcba", "GGAL")])

    assert set(result) == {("bcba", "GGAL")}
    quote = result[("bcba", "GGAL")]
    assert quote["last"] == pytest.approx(123.45)
    assert quote["chg_pct"] == pytest.approx(1.2)
    assert quote["asof"] == "2024-01-01T00:00:00"
    assert quote["provider"] == "iol"
    assert quote["stale"] is True
    assert recorded == [("iol", {"elapsed_ms": None, "stale": True, "source": "persistent"})]


def test_fetch_quotes_bulk_persists_and_rehydrates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_st = SimpleNamespace(session_state={})
    monkeypatch.setattr(cache_module, "st", fake_st)
    monkeypatch.setattr("shared.cache.st", fake_st, raising=False)

    recorded: list[tuple[str, dict[str, Any]]] = []

    def record(provider: str, **kwargs: Any) -> None:
        recorded.append((provider, kwargs))

    monkeypatch.setattr(cache_module, "record_quote_provider_usage", record)
    monkeypatch.setattr(cache_module, "record_quote_load", lambda *_, **__: None)
    monkeypatch.setattr(cache_module, "_persist_quote", ORIGINAL_PERSIST_QUOTE)
    monkeypatch.setattr(
        cache_module,
        "_QUOTE_PERSIST_PATH",
        tmp_path / "quotes_cache.json",
    )
    monkeypatch.setattr(cache_module, "QUOTE_STALE_TTL_SECONDS", 60.0)

    cache_module._QUOTE_PERSIST_CACHE = None
    cache_module._QUOTE_CACHE.clear()
    cache_module.fetch_quotes_bulk.clear()

    class DummyClient:
        def __init__(self, payloads: list[dict[tuple[str, str], dict[str, Any]]]) -> None:
            self._payloads = list(payloads)

        def get_quotes_bulk(self, items):  # type: ignore[no-untyped-def]
            if not self._payloads:
                pytest.fail("unexpected bulk call")
            return self._payloads.pop(0)

    client = DummyClient(
        [
            {
                ("bcba", "GGAL"): {
                    "last": 111.0,
                    "chg_pct": 1.1,
                    "asof": "2024-01-02T03:04:05",
                    "provider": "iol",
                }
            },
            {("bcba", "GGAL"): {"last": None, "provider": "iol"}},
        ]
    )

    try:
        first = cache_module.fetch_quotes_bulk(client, [("bcba", "GGAL")])
        assert set(first) == {("bcba", "GGAL")}
        quote = first[("bcba", "GGAL")]
        assert quote["last"] == pytest.approx(111.0)
        assert quote["provider"] == "iol"

        persist_key = cache_module._quote_cache_key("bcba", "GGAL", None)
        stored_entry = cache_module._load_persisted_entry(persist_key)
        assert stored_entry is not None
        stored_payload, _ = stored_entry
        assert stored_payload["last"] == pytest.approx(111.0)
        assert stored_payload["provider"] == "iol"
        assert stored_payload["asof"] == "2024-01-02T03:04:05"

        assert recorded
        first_provider, first_metrics = recorded[0]
        assert first_provider == "iol"
        assert first_metrics["stale"] is False
        assert first_metrics["source"] == "bulk"

        cache_module.fetch_quotes_bulk.clear()

        second = cache_module.fetch_quotes_bulk(client, [("bcba", "GGAL")])
        assert set(second) == {("bcba", "GGAL")}
        stale_quote = second[("bcba", "GGAL")]
        assert stale_quote["last"] == pytest.approx(111.0)
        assert stale_quote["stale"] is True
        assert stale_quote["provider"] == "iol"

        assert len(recorded) == 2
        fallback_provider, fallback_metrics = recorded[1]
        assert fallback_provider == "iol"
        assert fallback_metrics["stale"] is True
        assert fallback_metrics["source"] == "persistent"
    finally:
        cache_module.fetch_quotes_bulk.clear()
        cache_module._QUOTE_CACHE.clear()


def test_legacy_login_only_once_for_multiple_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = SimpleNamespace(session_state={})
    monkeypatch.setattr(iol_client_module, "st", fake_st)
    monkeypatch.setattr(cache_module, "st", fake_st)
    monkeypatch.setattr("shared.cache.st", fake_st, raising=False)

    class TrackingAuth:
        def __init__(self) -> None:
            self.tokens: dict[str, str] = {}
            self.login_calls = 0

        def login(self) -> dict[str, str]:
            self.login_calls += 1
            token = f"token-{self.login_calls}"
            self.tokens = {"access_token": token, "refresh_token": "refresh"}
            return self.tokens

        def auth_header(self) -> dict[str, str]:
            if not self.tokens.get("access_token"):
                self.login()
            return {"Authorization": f"Bearer {self.tokens['access_token']}"}

        def refresh(self) -> None:  # pragma: no cover - not exercised here
            raise AssertionError("refresh should not be called")

    auth = TrackingAuth()
    monkeypatch.setattr(iol_client_module.IOLClient, "_ensure_market_auth", lambda self: None)
    client = iol_client_module.IOLClient("user", "", auth=auth)

    def failing_request(self, method: str, url: str, **kwargs):  # type: ignore[no-untyped-def]
        raise _http_error(500)

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", failing_request)

    legacy_auths: list[TrackingAuth] = []

    class LegacyLoginSpy:
        def __init__(self, *args, **kwargs) -> None:
            self.auth = kwargs.get("auth")
            assert isinstance(self.auth, TrackingAuth)
            legacy_auths.append(self.auth)

        def get_quote(self, market, symbol, panel=None):  # type: ignore[no-untyped-def]
            header = self.auth.auth_header()
            assert header.get("Authorization")
            return {
                "last": 101.0,
                "chg_pct": 1.2,
                "asof": "2024-01-01T12:00:00",
                "provider": "legacy",
            }

    monkeypatch.setattr("infrastructure.iol.legacy.iol_client.IOLClient", LegacyLoginSpy)
    monkeypatch.setattr(ohlc_module, "OHLCAdapter", lambda *_, **__: pytest.fail("OHLC fallback should not run"))
    monkeypatch.setattr(cache_module, "_persist_quote", lambda *_, **__: None)
    monkeypatch.setattr(cache_module, "_load_persisted_entry", lambda key: None)
    monkeypatch.setattr(cache_module, "record_quote_provider_usage", lambda *_, **__: None)
    monkeypatch.setattr(cache_module, "record_quote_load", lambda *_, **__: None)
    monkeypatch.setattr(cache_module, "max_quote_workers", 1)
    cache_module.fetch_quotes_bulk.clear()
    cache_module._QUOTE_CACHE.clear()

    try:
        pairs = [("bcba", "GGAL"), ("nyse", "AAPL"), ("bcba", "YPFD")]
        result = cache_module.fetch_quotes_bulk(client, pairs)
    finally:
        cache_module._QUOTE_CACHE.clear()
        cache_module.fetch_quotes_bulk.clear()

    assert set(result) == {("bcba", "GGAL"), ("nyse", "AAPL"), ("bcba", "YPFD")}
    assert auth.login_calls == 1
    assert len({id(ref) for ref in legacy_auths}) == 1


def test_primary_client_uses_exponential_backoff_on_http_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = SimpleNamespace(session_state={})
    monkeypatch.setattr(iol_client_module, "st", fake_st)

    class StaticAuth:
        def __init__(self) -> None:
            self.tokens = {"access_token": "token", "refresh_token": "refresh"}

        def auth_header(self) -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        def refresh(self) -> None:  # pragma: no cover - not exercised
            raise AssertionError("refresh should not be triggered")

    monkeypatch.setattr(iol_client_module.IOLClient, "_ensure_market_auth", lambda self: None)
    client = iol_client_module.IOLClient("user", "", auth=StaticAuth())

    def failing_request(self, method: str, url: str, **kwargs):  # type: ignore[no-untyped-def]
        raise _http_error(429)

    monkeypatch.setattr(iol_client_module.requests.Session, "request", failing_request, raising=False)

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(iol_client_module.time, "sleep", fake_sleep)

    response = client._request("GET", "https://example.com/api")

    assert response is None
    assert sleeps == [0.5, 1.0, 2.0]
