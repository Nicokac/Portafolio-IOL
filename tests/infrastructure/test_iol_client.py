"""Tests for the native IOL client integration with bearer tokens."""

from __future__ import annotations

import sys
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import requests

from infrastructure.iol import client as iol_client_module
from tests.fixtures.auth import FakeAuth

# Ensure the project root is importable regardless of pytest's invocation path.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class StreamlitStub:
    """Streamlit stub exposing a mutable session_state mapping."""

    session_state: dict = {}


class StubIol:
    """Capture bearer assignment performed by the client."""

    def __init__(self, user: str, password: str) -> None:
        self.user = user
        self.password = password
        self.bearer: str | None = None
        self.refresh_token: str | None = None
        self.bearer_time: datetime | None = None
        self.gestionar_calls = 0

    def gestionar(self) -> None:
        self.gestionar_calls += 1


@pytest.fixture
def aware_moment() -> datetime:
    """Known aware timestamp used across the test."""

    return datetime(2024, 4, 1, 10, 45, tzinfo=ZoneInfo("UTC"))


@pytest.fixture(autouse=True)
def stub_record_usage(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Capture quote provider usage metrics emitted by the client."""

    calls: list[dict] = []

    def fake_record(
        provider: str,
        *,
        elapsed_ms: float | None,
        stale: bool,
        source: str | None = None,
        ok: bool | None = None,
    ) -> None:
        calls.append(
            {
                "provider": provider,
                "elapsed_ms": elapsed_ms,
                "stale": stale,
                "source": source,
            }
        )

    monkeypatch.setattr(iol_client_module, "record_quote_provider_usage", fake_record)
    return calls


def test_client_assigns_naive_bearer_time(monkeypatch: pytest.MonkeyPatch, aware_moment: datetime) -> None:
    """Instantiating IOLClient with bearer tokens should set a naive bearer_time."""

    call_count = 0

    def fake_now_datetime(cls: type[iol_client_module.TimeProvider]) -> datetime:
        nonlocal call_count
        call_count += 1
        return aware_moment

    monkeypatch.setattr(
        iol_client_module.TimeProvider,
        "now_datetime",
        classmethod(fake_now_datetime),
    )
    monkeypatch.setattr(iol_client_module, "Iol", StubIol)
    monkeypatch.setattr(iol_client_module, "st", StreamlitStub)

    client = iol_client_module.IOLClient("user", "", auth=FakeAuth(access="access", refresh="refresh"))

    assert isinstance(client.iol_market, StubIol)
    assert client.iol_market.bearer == "access"
    assert client.iol_market.refresh_token == "refresh"
    assert client.iol_market.bearer_time is not None
    assert client.iol_market.bearer_time.tzinfo is None
    assert client.iol_market.bearer_time == aware_moment.replace(tzinfo=None)
    assert call_count == 1


def test_get_quote_returns_last_and_chg_pct(monkeypatch: pytest.MonkeyPatch, aware_moment: datetime) -> None:
    """Quick hotfix validation: pytest tests/infrastructure/test_iol_client.py -k get_quote."""

    class ResponseStub:
        def json(self) -> dict:
            return {
                "simbolo": "AAPL",
                "ultimoPrecio": 123.45,
                "variacion": 1.5,
                "moneda": "ARS",
            }

    def fake_request(self: iol_client_module.IOLClient, method: str, url: str, **kwargs) -> ResponseStub:
        return ResponseStub()

    def fake_now_datetime(cls: type[iol_client_module.TimeProvider]) -> datetime:
        return aware_moment

    monkeypatch.setattr(
        iol_client_module.TimeProvider,
        "now_datetime",
        classmethod(fake_now_datetime),
    )
    monkeypatch.setattr(iol_client_module, "Iol", StubIol)
    monkeypatch.setattr(iol_client_module, "st", StreamlitStub)
    monkeypatch.setattr(iol_client_module.IOLClient, "_request", fake_request)

    client = iol_client_module.IOLClient("user", "", auth=FakeAuth(access="access", refresh="refresh"))

    result = client.get_quote("bcba", "AAPL")

    assert result["last"] == 123.45
    assert result["chg_pct"] == 1.5
    assert result["asof"] is None
    assert result["provider"] == "iol"
    assert result["proveedor_original"] == "iol"
    assert result["moneda_origen"] == "ARS"
    assert result.get("currency") == "ARS"
    assert result["fx_aplicado"] is None


def test_get_quote_returns_valid_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_quote should request the Cotizacion endpoint and parse the payload."""

    calls: dict[str, str] = {}

    monkeypatch.setattr(iol_client_module.IOLClient, "_ensure_market_auth", lambda self: None)

    def fake_request(self: iol_client_module.IOLClient, method: str, url: str, **kwargs) -> SimpleNamespace:
        calls["method"] = method
        calls["url"] = url
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {
                "ultimoPrecio": 100,
                "variacionPorcentual": 1.23,
                "moneda": "USD",
            },
        )

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", fake_request)

    client = iol_client_module.IOLClient("", "", auth=False)

    payload = client.get_quote("bcba", "AAPL")

    assert calls["url"].endswith("/bcba/Titulos/AAPL/Cotizacion")
    assert payload["last"] == 100.0
    assert payload["chg_pct"] == 1.23
    assert payload["asof"] is None
    assert payload["provider"] == "iol"
    assert payload["proveedor_original"] == "iol"
    assert payload["moneda_origen"] == "USD"
    assert payload.get("currency") == "USD"


def test_get_quote_passes_panel_query_param(monkeypatch: pytest.MonkeyPatch) -> None:
    """The primary endpoint should receive the panel as a query parameter."""

    calls: dict[str, Any] = {}

    monkeypatch.setattr(iol_client_module.IOLClient, "_ensure_market_auth", lambda self: None)

    def fake_request(self: iol_client_module.IOLClient, method: str, url: str, **kwargs) -> SimpleNamespace:
        calls["params"] = kwargs.get("params")
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"ultimoPrecio": 90, "variacion": 0.5, "moneda": "ARS"},
        )

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", fake_request)

    client = iol_client_module.IOLClient("", "", auth=False)

    payload = client.get_quote("bcba", "GGAL", panel="PanelGeneral")

    assert calls["params"] == {"panel": "PanelGeneral"}
    assert payload["provider"] == "iol"
    assert payload["proveedor_original"] == "iol"
    assert payload["moneda_origen"] == "ARS"
    assert payload["currency"] == "ARS"


def test_get_quote_falls_back_to_ohlc_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """When v2 and legacy fail, the OHLC adapter should provide a payload."""

    class LegacyStubClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_quote(self, **kwargs):
            raise RuntimeError("legacy boom")

    fallback_payload = {
        "last": 150.0,
        "chg_pct": 1.1,
        "asof": "2024-01-02T12:00:00",
        "provider": "av",
    }

    legacy_module = types.SimpleNamespace(IOLClient=LegacyStubClient)
    monkeypatch.setitem(sys.modules, "infrastructure.iol.compat.iol_client", legacy_module)
    monkeypatch.setattr(iol_client_module.IOLClient, "_ensure_market_auth", lambda self: None)
    monkeypatch.setattr(
        iol_client_module.IOLClient,
        "_request",
        lambda *_, **__: (_ for _ in ()).throw(requests.HTTPError(response=SimpleNamespace(status_code=500))),
    )
    monkeypatch.setattr(
        iol_client_module.IOLClient,
        "_fallback_quote_via_ohlc",
        lambda self, market, symbol, panel=None: fallback_payload,
    )

    client = iol_client_module.IOLClient("user", "", auth=FakeAuth())
    result = client.get_quote("bcba", "GGAL")

    assert result == fallback_payload


def test_get_quote_marks_stale_when_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If no provider responds, the payload should be marked as stale."""

    monkeypatch.setattr(iol_client_module.IOLClient, "_ensure_market_auth", lambda self: None)
    monkeypatch.setattr(iol_client_module.IOLClient, "_request", lambda *_, **__: None)
    monkeypatch.setattr(
        iol_client_module.IOLClient,
        "_legacy_quote_fallback",
        lambda self, market, symbol, panel=None: (None, {}),
    )
    monkeypatch.setattr(
        iol_client_module.IOLClient,
        "_fallback_quote_via_ohlc",
        lambda self, market, symbol, panel=None: None,
    )

    client = iol_client_module.IOLClient("user", "", auth=FakeAuth())
    payload = client.get_quote("bcba", "GGAL")

    assert payload["last"] is None
    assert payload["chg_pct"] is None
    assert payload["asof"] is None
    assert payload["provider"] == "stale"
    assert payload["proveedor_original"] == "stale"
    assert payload["moneda_origen"] is None
    assert payload["fx_aplicado"] is None


def test_get_quote_response_none_uses_legacy(monkeypatch: pytest.MonkeyPatch, stub_record_usage: list[dict]) -> None:
    """When the primary request yields no response, the legacy fallback should supply the quote."""

    monkeypatch.setattr(iol_client_module.IOLClient, "_ensure_market_auth", lambda self: None)
    monkeypatch.setattr(iol_client_module.IOLClient, "_request", lambda *_, **__: None)

    def fake_legacy(
        self: iol_client_module.IOLClient,
        market: str,
        symbol: str,
        panel: str | None,
    ) -> tuple[dict, dict[str, bool]]:
        return (
            {
                "last": 250.5,
                "chg_pct": 2.1,
                "provider": "legacy",
                "proveedor_original": "legacy",
                "moneda_origen": "ARS",
                "fx_aplicado": None,
            },
            {},
        )

    monkeypatch.setattr(
        iol_client_module.IOLClient,
        "_legacy_quote_fallback",
        fake_legacy,
    )

    def fail_ohlc(*args, **kwargs):
        raise AssertionError("OHLC fallback should not be used when legacy succeeds")

    monkeypatch.setattr(
        iol_client_module.IOLClient,
        "_fallback_quote_via_ohlc",
        fail_ohlc,
    )

    client = iol_client_module.IOLClient("user", "", auth=FakeAuth())
    payload = client.get_quote("bcba", "GGAL")

    assert payload["last"] == 250.5
    assert payload["chg_pct"] == 2.1
    assert payload["provider"] == "legacy"
    assert payload["proveedor_original"] == "legacy"
    assert payload["moneda_origen"] == "ARS"
    assert stub_record_usage == [
        {
            "provider": "iol",
            "elapsed_ms": None,
            "stale": True,
            "source": "v2-error",
        },
        {
            "provider": "legacy",
            "elapsed_ms": None,
            "stale": False,
            "source": "v2->legacy",
        },
    ]
