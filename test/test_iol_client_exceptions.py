from __future__ import annotations

import json

import pytest
import requests

from infrastructure.iol import client as client_module


class DummyAuth:
    def __init__(self) -> None:
        self.tokens = {"access_token": "tok", "refresh_token": "ref"}

    def auth_header(self) -> dict:  # pragma: no cover - unused in this test
        return {"Authorization": "Bearer tok"}

    def refresh(self) -> None:  # pragma: no cover - unused in this test
        raise AssertionError


def test_get_portfolio_uses_cache_on_network_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    cache = tmp_path / "portfolio.json"
    cache.write_text(json.dumps({"activos": [1]}), encoding="utf-8")
    monkeypatch.setattr("infrastructure.iol.client.PORTFOLIO_CACHE", cache)
    monkeypatch.setattr(client_module.IOLClient, "_ensure_market_auth", lambda self: None, raising=False)

    client = client_module.IOLClient("u", "p", auth=DummyAuth())

    def fail():
        raise requests.RequestException("boom")

    monkeypatch.setattr(client, "_fetch_portfolio_live", fail, raising=False)
    assert client.get_portfolio() == {"activos": [1]}


def test_get_portfolio_propagates_unexpected(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("infrastructure.iol.client.PORTFOLIO_CACHE", tmp_path / "portfolio.json")
    monkeypatch.setattr(client_module.IOLClient, "_ensure_market_auth", lambda self: None, raising=False)
    client = client_module.IOLClient("u", "p", auth=DummyAuth())

    def unexpected():
        raise ValueError("unexpected")

    monkeypatch.setattr(client, "_fetch_portfolio_live", unexpected, raising=False)
    with pytest.raises(ValueError):
        client.get_portfolio()
