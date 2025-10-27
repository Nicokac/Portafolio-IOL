from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from infrastructure.iol import client as iol_client_module
from tests.fixtures.auth import FakeAuth


class _ResponseStub:
    def __init__(self, payload: Any, headers: dict[str, Any] | None = None) -> None:
        self._payload = payload
        self.status_code = 200
        self.request = SimpleNamespace(headers=headers or {})

    def json(self) -> Any:
        return self._payload


def test_get_raw_portfolio_returns_snapshot_with_meta(
    restore_real_iol_client,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(iol_client_module.LegacySession, "get", classmethod(lambda cls: None))
    monkeypatch.setattr(iol_client_module.IOLClient, "_ensure_market_auth", lambda self: None)
    payload = {"activos": [{"simbolo": "BPOC7", "cantidad": 10}]}
    response = _ResponseStub(payload, headers={"Authorization": "Bearer SECRET"})

    def fake_request(self: iol_client_module.IOLClient, method: str, url: str, **_: Any) -> _ResponseStub:
        assert method == "GET"
        assert url.endswith("/portafolio/argentina")
        return response

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", fake_request)
    client = iol_client_module.IOLClient("user", "pass", auth=FakeAuth())

    with caplog.at_level(logging.DEBUG):
        result = client.get_raw_portfolio()

    assert result is not payload
    assert result["activos"] == payload["activos"]
    assert "_meta" in result
    meta = result["_meta"]
    assert meta["endpoint"].endswith("/portafolio/argentina")
    assert meta["request_id"]
    assert "SECRET" not in caplog.text
    records = [record for record in caplog.records if record.message == "IOLClient raw fetch completed"]
    assert records
    assert getattr(records[0], "request_headers", {}).get("Authorization") == "***REDACTED***"
    assert "_meta" not in payload


def test_get_raw_quote_switches_endpoint(restore_real_iol_client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(iol_client_module.LegacySession, "get", classmethod(lambda cls: None))
    monkeypatch.setattr(iol_client_module.IOLClient, "_ensure_market_auth", lambda self: None)
    quote_payload = {"ultimoPrecio": 123.45}
    detail_payload = {"detalle": True}
    responses = {
        "Cotizacion": _ResponseStub(quote_payload),
        "CotizacionDetalle": _ResponseStub(detail_payload),
    }

    def fake_request(self: iol_client_module.IOLClient, method: str, url: str, **_: Any) -> _ResponseStub:
        assert method == "GET"
        if url.endswith("CotizacionDetalle"):
            return responses["CotizacionDetalle"]
        return responses["Cotizacion"]

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", fake_request)
    client = iol_client_module.IOLClient("user", "pass", auth=FakeAuth())

    quote = client.get_raw_quote(mercado="bcba", simbolo="BPOC7")
    detail = client.get_raw_quote(mercado="bcba", simbolo="BPOC7", detalle=True)

    assert quote is not quote_payload
    assert detail is not detail_payload
    assert quote["_meta"]["endpoint"].endswith("Cotizacion")
    assert detail["_meta"]["endpoint"].endswith("CotizacionDetalle")
    assert "_meta" not in quote_payload
    assert "_meta" not in detail_payload
