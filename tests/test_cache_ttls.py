"""Tests ensuring cache decorators use patched TTL values from settings."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


@contextmanager
def reload_ta_with(monkeypatch, setting_name: str, value: int):
    """Reload ``application.ta_service`` with a patched TTL setting value."""

    import application.ta_service as ta_module

    with monkeypatch.context() as mp:
        mp.setattr(f"shared.settings.{setting_name}", value)
        module = importlib.reload(ta_module)
        try:
            yield module, mp
        finally:
            for attr in (
                "fetch_with_indicators",
                "get_fundamental_data",
                "portfolio_fundamentals",
                "get_portfolio_history",
            ):
                func = getattr(module, attr, None)
                if func and hasattr(func, "clear"):
                    func.clear()

    importlib.reload(ta_module)


def test_fetch_portfolio_respects_monkeypatched_ttl(monkeypatch):
    """``fetch_portfolio`` should honour TTL overrides from ``shared.settings``."""

    with reload_cache_with(monkeypatch, "cache_ttl_portfolio", 0) as (cache_module, mp):
        mp.setattr(cache_module, "record_portfolio_load", lambda *_, **__: None)

        class DummyClient:
            def __init__(self) -> None:
                self.calls = 0
                self.auth = SimpleNamespace(tokens_path=None)

            def get_portfolio(self, country="argentina"):
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


def test_fetch_with_indicators_respects_monkeypatched_ttl(monkeypatch):
    """``fetch_with_indicators`` should honour Yahoo quotes TTL overrides."""

    with reload_ta_with(monkeypatch, "yahoo_quotes_ttl", 0) as (ta_module, mp):
        mp.setattr(ta_module, "map_to_us_ticker", lambda sym: sym)

        class DummyAdapter:
            def __init__(self) -> None:
                self.calls = 0

            def fetch(self, symbol, **kwargs):
                self.calls += 1
                idx = pd.date_range("2021-01-01", periods=60, freq="D")
                values = list(range(60))
                return pd.DataFrame(
                    {
                        "Open": values,
                        "High": values,
                        "Low": values,
                        "Close": values,
                        "Volume": [1] * 60,
                    },
                    index=idx,
                )

        dummy_adapter = DummyAdapter()
        mp.setattr(ta_module, "get_ohlc_adapter", lambda: dummy_adapter)

        class DummyRSI:
            def __init__(self, close, window, fillna):
                self._index = close.index

            def rsi(self):
                return pd.Series([50.0] * len(self._index), index=self._index)

        class DummyMACD:
            def __init__(self, close, window_slow, window_fast, window_sign):
                self._index = close.index

            def macd(self):
                return pd.Series([0.0] * len(self._index), index=self._index)

            def macd_signal(self):
                return pd.Series([0.0] * len(self._index), index=self._index)

            def macd_diff(self):
                return pd.Series([0.0] * len(self._index), index=self._index)

        class DummyATR:
            def __init__(self, high, low, close, window):
                self._index = close.index

            def average_true_range(self):
                return pd.Series([0.0] * len(self._index), index=self._index)

        class DummyStoch:
            def __init__(self, high, low, close, window, smooth_window):
                self._index = close.index

            def stoch(self):
                return pd.Series([0.0] * len(self._index), index=self._index)

            def stoch_signal(self):
                return pd.Series([0.0] * len(self._index), index=self._index)

        class DummyIchimoku:
            def __init__(self, high, low, window1, window2, window3):
                self._index = high.index

            def ichimoku_conversion_line(self):
                return pd.Series([0.0] * len(self._index), index=self._index)

            def ichimoku_base_line(self):
                return pd.Series([0.0] * len(self._index), index=self._index)

            def ichimoku_a(self):
                return pd.Series([0.0] * len(self._index), index=self._index)

            def ichimoku_b(self):
                return pd.Series([0.0] * len(self._index), index=self._index)

        mp.setattr(ta_module, "RSIIndicator", DummyRSI)
        mp.setattr(ta_module, "MACD", DummyMACD)
        mp.setattr(ta_module, "AverageTrueRange", DummyATR)
        mp.setattr(ta_module, "StochasticOscillator", DummyStoch)
        mp.setattr(ta_module, "IchimokuIndicator", DummyIchimoku)

        ta_module.fetch_with_indicators.clear()
        first = ta_module.fetch_with_indicators("AAPL")
        second = ta_module.fetch_with_indicators("AAPL")

        assert dummy_adapter.calls == 2
        assert not first.empty
        assert not second.empty


def test_get_fundamental_data_respects_monkeypatched_ttl(monkeypatch):
    """``get_fundamental_data`` should honour Yahoo fundamentals TTL overrides."""

    with reload_ta_with(monkeypatch, "yahoo_fundamentals_ttl", 0) as (ta_module, mp):
        class DummyYF:
            def __init__(self) -> None:
                self.calls = 0

            def Ticker(self, ticker):
                self.calls += 1
                return SimpleNamespace(
                    info={
                        "marketCap": 1,
                        "shortName": ticker,
                        "sector": "Tech",
                        "previousClose": 10,
                        "dividendRate": 0.5,
                    }
                )

        dummy_yf = DummyYF()
        mp.setattr(ta_module, "yf", dummy_yf)

        result_one = ta_module.get_fundamental_data("AAPL")
        result_two = ta_module.get_fundamental_data("AAPL")

        assert dummy_yf.calls == 2
        assert result_one["name"] == "AAPL"
        assert result_two["name"] == "AAPL"
