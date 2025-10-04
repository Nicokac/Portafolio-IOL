"""Tests for the native IOL client integration with bearer tokens."""

from __future__ import annotations

import sys
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import requests

# Ensure the project root is importable regardless of pytest's invocation path.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infrastructure.iol import client as iol_client_module
from infrastructure.iol.legacy import iol_client as legacy_module


class FakeAuth:
    """Minimal auth stub exposing preloaded tokens."""

    def __init__(self) -> None:
        self.tokens = {
            "access_token": "access",
            "refresh_token": "refresh",
        }

    def auth_header(self) -> dict:
        raise AssertionError("auth_header should not be called in this test")

    def refresh(self) -> None:
        raise AssertionError("refresh should not be called in this test")


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

    client = iol_client_module.IOLClient("user", "", auth=FakeAuth())

    assert isinstance(client.iol_market, StubIol)
    assert client.iol_market.bearer == "access"
    assert client.iol_market.refresh_token == "refresh"
    assert client.iol_market.bearer_time is not None
    assert client.iol_market.bearer_time.tzinfo is None
    assert client.iol_market.bearer_time == aware_moment.replace(tzinfo=None)
    assert call_count == 1


def test_get_quote_returns_last_and_chg_pct(
    monkeypatch: pytest.MonkeyPatch, aware_moment: datetime
) -> None:
    """Quick hotfix validation: pytest tests/infrastructure/test_iol_client.py -k get_quote."""

    class ResponseStub:
        def json(self) -> dict:
            return {"simbolo": "AAPL", "ultimoPrecio": 123.45, "variacion": 1.5}

    def fake_request(
        self: iol_client_module.IOLClient, method: str, url: str, **kwargs
    ) -> ResponseStub:
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

    client = iol_client_module.IOLClient("user", "", auth=FakeAuth())

    result = client.get_quote("bcba", "AAPL")

    assert result == {"last": 123.45, "chg_pct": 1.5}


def test_get_quote_returns_valid_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_quote should request the Cotizacion endpoint and parse the payload."""

    calls: dict[str, str] = {}

    monkeypatch.setattr(iol_client_module.IOLClient, "_ensure_market_auth", lambda self: None)

    def fake_request(
        self: iol_client_module.IOLClient, method: str, url: str, **kwargs
    ) -> SimpleNamespace:
        calls["method"] = method
        calls["url"] = url
        return SimpleNamespace(
            status_code=200,
            raise_for_status=lambda: None,
            json=lambda: {"ultimoPrecio": 100, "variacionPorcentual": 1.23},
        )

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", fake_request)

    client = iol_client_module.IOLClient("", "", auth=False)

    payload = client.get_quote("bcba", "AAPL")

    assert calls["url"].endswith("/Cotizacion")
    assert payload == {"last": 100.0, "chg_pct": 1.23}


def test_get_quote_falls_back_to_legacy_on_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If _request returns None the client should use the legacy fallback."""

    legacy_calls: dict[str, tuple[str, str, str | None]] = {}

    class LegacyStub:
        def __init__(self, *_: object, **__: object) -> None:
            legacy_calls["init"] = ("ok",)

        def get_quote(
            self, *, market: str, symbol: str, panel: str | None = None
        ) -> dict[str, float]:
            legacy_calls["args"] = (market, symbol, panel)
            return {"last": 10.0, "chg_pct": 2.5}

    monkeypatch.setattr(iol_client_module, "Iol", StubIol)
    monkeypatch.setattr(iol_client_module, "st", StreamlitStub)
    monkeypatch.setattr(iol_client_module.IOLClient, "_request", lambda *_, **__: None)
    monkeypatch.setattr(legacy_module, "IOLClient", LegacyStub)

    client = iol_client_module.IOLClient("user", "", auth=FakeAuth())

    payload = client.get_quote("bcba", "GGAL")

    assert payload == {"last": 10.0, "chg_pct": 2.5}
    assert legacy_calls["args"] == ("bcba", "GGAL", None)


def test_get_quotes_bulk_never_returns_none_with_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """500 errors should trigger the legacy fallback and return a neutral payload."""

    response = requests.Response()
    response.status_code = 500

    def failing_request(*_: object, **__: object) -> None:
        raise requests.HTTPError(response=response)

    class LegacyStub:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def get_quote(
            self, *, market: str, symbol: str, panel: str | None = None
        ) -> None:
            # Simulate the legacy client not having data for the requested symbol
            return None

    monkeypatch.setattr(iol_client_module, "Iol", StubIol)
    monkeypatch.setattr(iol_client_module, "st", StreamlitStub)
    monkeypatch.setattr(iol_client_module.IOLClient, "_request", failing_request)
    monkeypatch.setattr(legacy_module, "IOLClient", LegacyStub)

    client = iol_client_module.IOLClient("user", "", auth=FakeAuth())

    result = client.get_quotes_bulk([("bcba", "GGAL")], max_workers=1)

    assert result == {("bcba", "GGAL"): {"last": None, "chg_pct": None}}
