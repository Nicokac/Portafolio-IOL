import logging

import pytest

from infrastructure.iol import client as client_module
from infrastructure.iol.legacy import iol_client as legacy_module


class DummyLegacyClient:
    def __init__(self, *args, **kwargs):
        pass

    def get_portfolio(self):
        raise Exception("network error")


def test_adapter_get_portfolio_logs(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(client_module, "_LegacyIOLClient", DummyLegacyClient)
    monkeypatch.setattr(client_module, "PORTFOLIO_CACHE", tmp_path / "portfolio.json")

    adapter = client_module.IOLClientAdapter("user", "pass")

    with caplog.at_level(logging.WARNING):
        result = adapter.get_portfolio()

    assert result == {"activos": []}
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("get_portfolio falló" in r.message for r in warnings)
    assert any("No se pudo leer cache portafolio" in r.message for r in errors)


class DummyMarket:
    def price_to_json(self, *args, **kwargs):
        raise Exception("network fail")


def fake_ensure_market_auth(self):
    self.iol_market = DummyMarket()
    self._market_ready = True


def test_legacy_get_last_price_logs(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(legacy_module.IOLClient, "_ensure_market_auth", fake_ensure_market_auth, raising=False)
    client = legacy_module.IOLClient("u", "p", tokens_file=tmp_path / "tokens.json")

    with caplog.at_level(logging.WARNING):
        price = client.get_last_price("m", "SYM")

    assert price is None
    assert any("get_last_price error" in r.message for r in caplog.records)
