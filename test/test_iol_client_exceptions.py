import json
import pytest
import requests

from infrastructure.iol import client as client_module


class DummyLegacyClientError:
    def __init__(self, *args, **kwargs):
        pass

    def get_portfolio(self):
        raise requests.RequestException("boom")


class DummyLegacyClientUnexpected:
    def __init__(self, *args, **kwargs):
        pass

    def get_portfolio(self):
        raise ValueError("unexpected")


def test_get_portfolio_uses_cache_on_network_error(monkeypatch, tmp_path):
    monkeypatch.setattr(client_module, "_LegacyIOLClient", DummyLegacyClientError)
    cache = tmp_path / "portfolio.json"
    cache.write_text(json.dumps({"activos": [1]}), encoding="utf-8")
    monkeypatch.setattr(client_module, "PORTFOLIO_CACHE", cache)

    adapter = client_module.IOLClientAdapter("u", "p")
    assert adapter.get_portfolio() == {"activos": [1]}


def test_get_portfolio_propagates_unexpected(monkeypatch, tmp_path):
    monkeypatch.setattr(client_module, "_LegacyIOLClient", DummyLegacyClientUnexpected)
    monkeypatch.setattr(client_module, "PORTFOLIO_CACHE", tmp_path / "portfolio.json")
    adapter = client_module.IOLClientAdapter("u", "p")
    with pytest.raises(ValueError):
        adapter.get_portfolio()
