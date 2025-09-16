"""Tests ensuring cache decorators use patched TTL values from settings."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from types import SimpleNamespace


@contextmanager
def reload_cache_with(monkeypatch, setting_name: str, value: int):
    """Reload ``services.cache`` after patching a TTL setting value."""

    import services.cache as cache_module

    with monkeypatch.context() as mp:
        mp.setattr(f"shared.settings.{setting_name}", value)
        module = importlib.reload(cache_module)
        try:
            yield module, mp
        finally:
            for attr in ("fetch_portfolio", "fetch_quotes_bulk", "fetch_fx_rates"):
                func = getattr(module, attr, None)
                if func and hasattr(func, "clear"):
                    func.clear()

    importlib.reload(cache_module)


def test_fetch_portfolio_respects_monkeypatched_ttl(monkeypatch):
    """``fetch_portfolio`` should honour TTL overrides from ``shared.settings``."""

    with reload_cache_with(monkeypatch, "cache_ttl_portfolio", 0) as (cache_module, mp):
        mp.setattr(cache_module, "record_portfolio_load", lambda *_, **__: None)

        class DummyClient:
            def __init__(self) -> None:
                self.calls = 0
                self.auth = SimpleNamespace(tokens_path=None)

            def get_portfolio(self):
                self.calls += 1
                return {"calls": self.calls}

        client = DummyClient()

        cache_module.fetch_portfolio.clear()
        first = cache_module.fetch_portfolio(client)
        second = cache_module.fetch_portfolio(client)

        assert client.calls == 2
        assert first == {"calls": 1}
        assert second == {"calls": 2}


def test_fetch_quotes_bulk_respects_monkeypatched_ttl(monkeypatch):
    """``fetch_quotes_bulk`` should honour TTL overrides from ``shared.settings``."""

    with reload_cache_with(monkeypatch, "cache_ttl_quotes", 0) as (cache_module, mp):
        mp.setattr(cache_module, "record_quote_load", lambda *_, **__: None)

        class DummyClient:
            def __init__(self) -> None:
                self.calls = 0

            def get_quotes_bulk(self, items):
                self.calls += 1
                return {
                    tuple(item): {"last": float(self.calls), "chg_pct": float(self.calls)}
                    for item in items
                }

        client = DummyClient()
        items = [("bcba", "GGAL")]

        cache_module.fetch_quotes_bulk.clear()
        first = cache_module.fetch_quotes_bulk(client, items)
        second = cache_module.fetch_quotes_bulk(client, items)

        assert client.calls == 2
        assert first[items[0]]["last"] == 1.0
        assert second[items[0]]["last"] == 2.0


def test_fetch_fx_rates_respects_monkeypatched_ttl(monkeypatch):
    """``fetch_fx_rates`` should honour TTL overrides from ``shared.settings``."""

    with reload_cache_with(monkeypatch, "cache_ttl_fx", 0) as (cache_module, mp):
        mp.setattr(cache_module, "record_fx_api_response", lambda *_, **__: None)

        class DummyProvider:
            def __init__(self) -> None:
                self.calls = 0

            def get_rates(self):
                self.calls += 1
                return {"USD": self.calls}, None

            def close(self):
                pass

        provider = DummyProvider()
        mp.setattr(cache_module, "get_fx_provider", lambda: provider)

        cache_module.fetch_fx_rates.clear()
        first = cache_module.fetch_fx_rates()
        second = cache_module.fetch_fx_rates()

        assert provider.calls == 2
        assert first[0] == {"USD": 1}
        assert second[0] == {"USD": 2}
