from __future__ import annotations

import json

import pytest
import requests

from infrastructure.iol.account_client import AccountCashSummary, IOLAccountClient


class DummyAuth:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def ensure_token(self, *, silent: bool = True):  # noqa: ANN001 - mimic signature
        self.calls.append("ensure")

    def auth_header(self) -> dict[str, str]:
        return {"Authorization": "Bearer token"}

    def refresh(self, *, silent: bool = True):  # noqa: ANN001 - mimic signature
        self.calls.append("refresh")


def test_account_client_parses_balances():
    auth = DummyAuth()
    client = IOLAccountClient(auth=auth, session=None, api_base="https://api.example.com/api/v2")

    payload = {
        "cotizacionDolar": 500.0,
        "disponibleEnPesos": 1200.0,
        "disponibleEnDolares": 10.0,
        "cuentas": [
            {"moneda": "ARS", "disponible": 300.0},
            {"moneda": "USD", "disponible": 5.0, "cotizacionCartera": 510.0},
        ],
    }

    summary = client._parse_payload(payload)  # noqa: SLF001 - exercised for coverage

    assert isinstance(summary, AccountCashSummary)
    assert summary.cash_ars == pytest.approx(1500.0)
    assert summary.cash_usd == pytest.approx(15.0)
    assert summary.usd_rate == pytest.approx(500.0)
    assert summary.to_payload()["cash_usd_ars_equivalent"] == pytest.approx(7500.0)


def test_account_client_refreshes_on_401(monkeypatch):
    auth = DummyAuth()
    responses: list[requests.Response] = []

    def _response(status_code: int, payload: dict[str, object]) -> requests.Response:
        resp = requests.Response()
        resp.status_code = status_code
        resp._content = json.dumps(payload).encode()
        resp.headers["Content-Type"] = "application/json"
        resp.url = "https://api.example.com/api/v2/estadocuenta"
        return resp

    responses.append(_response(401, {"error": "unauthorized"}))
    responses.append(
        _response(
            200,
            {
                "disponibleEnPesos": 0,
                "cuentas": [
                    {"moneda": "USD", "disponible": 2.0, "cotizacion": 480.0},
                ],
            },
        )
    )

    def fake_request(self, method, url, **kwargs):  # noqa: ANN001 - match requests.Session.request signature
        if not responses:
            raise AssertionError("request called more times than expected")
        resp = responses.pop(0)
        return resp

    client = IOLAccountClient(auth=auth, session=None, api_base="https://api.example.com/api/v2")

    monkeypatch.setattr(type(client._session), "request", fake_request, raising=False)

    summary = client.fetch_balances()

    assert "ensure" in auth.calls
    assert "refresh" in auth.calls
    assert summary.cash_usd == pytest.approx(2.0)
