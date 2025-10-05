"""Logging behaviour for the native IOL client."""

from __future__ import annotations

import importlib
import logging

import pytest
import requests

from infrastructure.iol import client as client_module


class DummyAuth:
    def __init__(self) -> None:
        self.tokens = {"access_token": "tok", "refresh_token": "ref"}

    def auth_header(self) -> dict:
        return {"Authorization": "Bearer tok"}

    def refresh(self) -> None:  # pragma: no cover - refresh not expected
        raise AssertionError("refresh should not be called")


def test_client_get_portfolio_logs(monkeypatch: pytest.MonkeyPatch, tmp_path, caplog) -> None:
    monkeypatch.setattr(client_module.IOLClient, "_ensure_market_auth", lambda self: None, raising=False)
    monkeypatch.setattr("infrastructure.iol.client.PORTFOLIO_CACHE", tmp_path / "portfolio.json")

    def fail(self, country="argentina"):
        raise requests.RequestException("network error")

    monkeypatch.setattr(client_module.IOLClient, "_fetch_portfolio_live", fail, raising=False)

    client = client_module.IOLClient("user", "pass", auth=DummyAuth())

    with caplog.at_level(logging.WARNING):
        result = client.get_portfolio()

    assert result == {"activos": []}
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert any("get_portfolio falló" in record.message for record in warnings)
    assert any("No se pudo leer cache portafolio" in record.message for record in errors)


class DummyMarket:
    def price_to_json(self, *args, **kwargs):  # pragma: no cover - used in tests only
        raise Exception("network fail")


def fake_ensure_market_auth(self):
    self.iol_market = DummyMarket()
    self._market_ready = True


def test_client_get_last_price_logs(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    import infrastructure.iol.auth as auth_module

    importlib.reload(auth_module)
    monkeypatch.setattr(client_module.IOLClient, "_ensure_market_auth", fake_ensure_market_auth, raising=False)
    client = client_module.IOLClient("u", "p", auth=DummyAuth())

    with caplog.at_level(logging.WARNING):
        price = client.get_last_price(mercado="m", simbolo="SYM")

    assert price is None
    assert any("get_last_price error" in record.message for record in caplog.records)

