import json

import numpy as np
import pandas as pd
import pytest

from application.screener import opportunities as ops
from controllers.opportunities import run_opportunities_controller


class AutoYahooClient:
    def __init__(self, data: dict[str, dict[str, object]]) -> None:
        self._data = data

    def get_fundamentals(self, ticker: str) -> dict[str, object]:
        return self._data[ticker]["fundamentals"]

    def get_dividends(self, ticker: str) -> pd.DataFrame:
        return self._data[ticker]["dividends"]

    def get_shares_outstanding(self, ticker: str) -> pd.DataFrame:
        return self._data[ticker]["shares"]

    def get_price_history(self, ticker: str) -> pd.DataFrame:
        return self._data[ticker]["prices"]


@pytest.fixture
def auto_dataset() -> dict[str, dict[str, object]]:
    dates = pd.date_range("2020-01-01", periods=365, freq="D", tz="UTC")
    growth = 1.05 ** (np.arange(len(dates)) / 365)
    prices = pd.DataFrame(
        {
            "date": dates,
            "close": 100 * growth,
            "adj_close": 100 * growth,
            "volume": np.linspace(1_000, 2_000, len(dates)),
        }
    )

    dividends = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-03-01", "2022-03-01", "2023-03-01"], utc=True),
            "amount": [1.0, 1.1, 1.2],
        }
    )

    shares = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2022-01-01", "2023-01-01"], utc=True),
            "shares": [1_000_000, 995_000, 990_000],
        }
    )

    base_fundamentals = {
        "dividend_yield": 2.0,
        "payout_ratio": 45.0,
    }

    tickers = ["AAA", "BBB", "CCC"]
    dataset: dict[str, dict[str, object]] = {}
    for idx, ticker in enumerate(tickers, start=1):
        fundamentals = dict(base_fundamentals)
        fundamentals["ticker"] = ticker
        fundamentals["dividend_yield"] += idx
        fundamentals["payout_ratio"] -= idx
        dataset[ticker] = {
            "fundamentals": fundamentals,
            "dividends": dividends,
            "shares": shares,
            "prices": prices,
        }
    return dataset


def test_controller_uses_auto_universe(monkeypatch, auto_dataset):
    monkeypatch.setattr(
        ops,
        "_get_symbol_pool",
        lambda: [
            {"ticker": "AAA", "market_cap": 6_000_000_000, "pe": 18.0, "revenue_growth": 12.0, "region": "US"},
            {"ticker": "BBB", "market_cap": 3_000_000_000, "pe": 35.0, "revenue_growth": 8.0, "region": "LATAM"},
            {"ticker": "CCC", "market_cap": 9_000_000_000, "pe": 20.0, "revenue_growth": 15.0, "region": "US"},
        ],
    )

    client = AutoYahooClient(auto_dataset)
    monkeypatch.setattr(ops, "YahooFinanceClient", lambda: client)

    df, notes, source = run_opportunities_controller(
        manual_tickers=None,
        include_technicals=False,
        min_market_cap=8_000_000_000,
        max_pe=25.0,
        min_revenue_growth=10.0,
        include_latam=False,
    )

    assert list(df["ticker"]) == ["CCC"]
    assert not df.isna().all(axis=None)
    assert any("seleccionados automáticamente" in note for note in notes)
    assert any("min_market_cap" in note for note in notes)
    assert source == "yahoo"


def test_controller_reports_when_no_candidates(monkeypatch):
    monkeypatch.setattr(ops, "_get_symbol_pool", lambda: [])
    monkeypatch.setattr(ops, "YahooFinanceClient", lambda: AutoYahooClient({}))

    df, notes, source = run_opportunities_controller(manual_tickers=None)

    assert df.empty
    assert any("No se encontraron símbolos" in note for note in notes)
    assert source == "yahoo"


def test_controller_uses_env_symbol_pool(monkeypatch, auto_dataset):
    env_pool = [
        {"ticker": "BBB", "market_cap": 3_000_000_000, "pe": 35.0, "revenue_growth": 18.0, "region": "LATAM"},
        {"ticker": "CCC", "market_cap": 9_000_000_000, "pe": 20.0, "revenue_growth": 15.0, "region": "US"},
        {"ticker": "AAA", "market_cap": 6_000_000_000, "pe": 18.0, "revenue_growth": 12.0, "region": "US"},
    ]
    monkeypatch.setenv("OPPORTUNITIES_SYMBOL_POOL", json.dumps(env_pool))
    monkeypatch.setattr(ops, "YahooFinanceClient", lambda: AutoYahooClient(auto_dataset))

    df, notes, source = run_opportunities_controller(
        manual_tickers=None,
        include_technicals=False,
        min_market_cap=8_000_000_000,
        max_pe=25.0,
        min_revenue_growth=10.0,
        include_latam=False,
    )

    assert list(df["ticker"]) == ["CCC"]
    assert source == "yahoo"
    assert any("seleccionados automáticamente" in note for note in notes)
