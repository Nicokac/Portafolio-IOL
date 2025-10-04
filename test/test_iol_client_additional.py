"""Additional coverage for the native IOL client helpers."""

from __future__ import annotations

import pytest
import requests

from infrastructure.iol import client as client_module
from iolConn.common.exceptions import NoAuthException


class DummyAuth:
    def __init__(self) -> None:
        self.refreshed = False
        self.tokens = {"access_token": "a", "refresh_token": "r"}

    def auth_header(self) -> dict:
        return {"Authorization": "Bearer tok"}

    def refresh(self) -> None:
        self.refreshed = True


def _noop_auth(self):
    self._market_ready = True


@pytest.mark.parametrize(
    "data,expected",
    [
        ({"ultimoPrecio": 100}, 100.0),
        ({"last": "123,45"}, 123.45),
        ({"ultimo": {"value": "77.7"}}, 77.7),
        ({"foo": "bar"}, None),
        ("notdict", None),
    ],
)
def test_parse_price_fields_various(data, expected):
    assert client_module.IOLClient._parse_price_fields(data) == expected


@pytest.mark.parametrize(
    "data,last,expected",
    [
        ({"variacion": "1,5%"}, None, 1.5),
        ({"changePercent": 2}, None, 2.0),
        ({"cierreAnterior": "100", "puntosVariacion": "5"}, None, 5.0),
        ({"cierreAnterior": 100, "ultimo": 110}, None, 10.0),
        ({"cierreAnterior": 100}, 110, 10.0),
        ({"foo": "bar"}, None, None),
        ("bad", None, None),
    ],
)
def test_parse_chg_pct_fields_various(data, last, expected):
    assert client_module.IOLClient._parse_chg_pct_fields(data, last) == expected


def test_request_refresh_on_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module.IOLClient, "_ensure_market_auth", _noop_auth, raising=False)
    client = client_module.IOLClient("u", "p", auth=DummyAuth())

    class DummyResp:
        def __init__(self, code):
            self.status_code = code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(response=self)

    responses = iter([DummyResp(401), DummyResp(200)])
    count = {"n": 0}

    def fake_request(self, method, url, headers=None, timeout=None, **kwargs):
        count["n"] += 1
        return next(responses)

    monkeypatch.setattr(requests.Session, "request", fake_request)

    resp = client._request("GET", "http://example.com")

    assert resp.status_code == 200
    assert client.auth.refreshed
    assert count["n"] == 2


def test_request_returns_none_on_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module.IOLClient, "_ensure_market_auth", _noop_auth, raising=False)
    client = client_module.IOLClient("u", "p", auth=DummyAuth())

    class DummyResp:
        status_code = 404

        def raise_for_status(self):
            raise requests.HTTPError(response=self)

    def fake_request(self, method, url, headers=None, timeout=None, **kwargs):
        return DummyResp()

    monkeypatch.setattr(requests.Session, "request", fake_request)

    resp = client._request("GET", "http://example.com")
    assert resp is None
    assert not client.auth.refreshed


def test_ensure_market_auth_noauth_both(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_st = type("ST", (), {"session_state": {}})()
    monkeypatch.setattr(client_module, "st", dummy_st)

    class DummyIol:
        def __init__(self, *args, **kwargs):
            pass

        def gestionar(self):
            raise NoAuthException("fail")

    monkeypatch.setattr(client_module, "Iol", DummyIol)

    orig_ensure = client_module.IOLClient._ensure_market_auth
    monkeypatch.setattr(client_module.IOLClient, "_ensure_market_auth", lambda self: None, raising=False)
    client = client_module.IOLClient("u", "p", auth=DummyAuth())
    monkeypatch.setattr(client_module.IOLClient, "_ensure_market_auth", orig_ensure, raising=False)

    client._market_ready = False
    with pytest.raises(NoAuthException):
        client._ensure_market_auth()
    assert not client._market_ready


def test_get_last_price_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyMarket:
        def price_to_json(self, *args, **kwargs):
            raise Exception("boom")

    def fake_ensure(self):
        self.iol_market = DummyMarket()
        self._market_ready = True

    monkeypatch.setattr(client_module.IOLClient, "_ensure_market_auth", fake_ensure, raising=False)
    client = client_module.IOLClient("u", "p", auth=DummyAuth())
    assert client.get_last_price(mercado="m", simbolo="s") is None


def test_get_quote_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyMarket:
        def price_to_json(self, *args, **kwargs):
            return "nondict"

    def fake_ensure(self):
        self.iol_market = DummyMarket()
        self._market_ready = True

    monkeypatch.setattr(client_module.IOLClient, "_ensure_market_auth", fake_ensure, raising=False)
    client = client_module.IOLClient("u", "p", auth=DummyAuth())
    quote = client.get_quote(mercado="m", simbolo="s")
    assert quote == {"last": None, "chg_pct": None}


def test_get_quotes_bulk_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module.IOLClient, "_ensure_market_auth", _noop_auth, raising=False)

    def bad_get_quote(self, m, s):
        raise ValueError("fail")

    monkeypatch.setattr(client_module.IOLClient, "get_quote", bad_get_quote, raising=False)
    client = client_module.IOLClient("u", "p", auth=DummyAuth())
    result = client.get_quotes_bulk([("m", "SYM")])
    assert result == {("m", "SYM"): {"last": None, "chg_pct": None}}

