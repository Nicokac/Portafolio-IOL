import json
import requests

from infrastructure.iol import client as iol_client


class DummyAuth:
    def __init__(self) -> None:
        self.tokens = {"access_token": "tok", "refresh_token": "ref"}

    def auth_header(self) -> dict:  # pragma: no cover - unused
        return {"Authorization": "Bearer tok"}


def test_portfolio_fallback(monkeypatch, tmp_path):
    cache_file = tmp_path / "cache.json"
    monkeypatch.setattr(iol_client, "PORTFOLIO_CACHE", cache_file)
    monkeypatch.setattr(iol_client.IOLClient, "_ensure_market_auth", lambda self: None, raising=False)

    client = iol_client.IOLClient("user", "pass", auth=DummyAuth())

    monkeypatch.setattr(client, "_fetch_portfolio_live", lambda: {"activos": [1]}, raising=False)
    assert client.get_portfolio() == {"activos": [1]}
    assert json.loads(cache_file.read_text()) == {"activos": [1]}

    def fail():
        raise requests.RequestException("boom")

    monkeypatch.setattr(client, "_fetch_portfolio_live", fail, raising=False)
    assert client.get_portfolio() == {"activos": [1]}

    cache_file.unlink()
    assert client.get_portfolio() == {"activos": []}
