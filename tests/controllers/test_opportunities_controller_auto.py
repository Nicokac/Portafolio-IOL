import numpy as np
import pandas as pd
import pytest

from application.screener import opportunities as ops
from controllers.opportunities import run_opportunities_controller


class AutoYahooClient:
    def __init__(
        self,
        data: dict[str, dict[str, object]],
        *,
        listings: dict[str, list[dict[str, object] | str]] | None = None,
    ) -> None:
        self._data = data
        self._listings = listings or {}

    def get_fundamentals(self, ticker: str) -> dict[str, object]:
        return self._data[ticker]["fundamentals"]

    def get_dividends(self, ticker: str) -> pd.DataFrame:
        return self._data[ticker]["dividends"]

    def get_shares_outstanding(self, ticker: str) -> pd.DataFrame:
        return self._data[ticker]["shares"]

    def get_price_history(self, ticker: str) -> pd.DataFrame:
        return self._data[ticker]["prices"]

    def list_symbols_by_markets(self, markets: list[str]) -> list[dict[str, object] | str]:
        results: list[dict[str, object] | str] = []
        for market in markets:
            entries = self._listings.get(market, [])
            for entry in entries:
                if isinstance(entry, dict):
                    results.append(dict(entry))
                else:
                    results.append(entry)
        return results


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
        "market_cap": 5_000_000_000,
        "pe_ratio": 20.0,
        "revenue_growth": 8.0,
        "country": "United States",
    }

    tickers = ["AAA", "BBB", "CCC"]
    dataset: dict[str, dict[str, object]] = {}
    for idx, ticker in enumerate(tickers, start=1):
        fundamentals = dict(base_fundamentals)
        fundamentals["ticker"] = ticker
        fundamentals["dividend_yield"] += idx
        fundamentals["payout_ratio"] -= idx
        fundamentals["market_cap"] += idx * 1_500_000_000
        fundamentals["pe_ratio"] += idx
        fundamentals["revenue_growth"] += idx * 2
        if ticker == "BBB":
            fundamentals["country"] = "Argentina"
        dataset[ticker] = {
            "fundamentals": fundamentals,
            "dividends": dividends,
            "shares": shares,
            "prices": prices,
        }
    return dataset


def test_controller_uses_auto_universe(monkeypatch, auto_dataset):
    listings = {"TEST": [{"ticker": "AAA"}, {"ticker": "BBB"}, {"ticker": "CCC"}]}

    client = AutoYahooClient(auto_dataset, listings=listings)
    monkeypatch.setattr(ops, "YahooFinanceClient", lambda: client)
    monkeypatch.setattr(ops, "_get_target_markets", lambda: ["TEST"])

    payload = run_opportunities_controller(
        manual_tickers=None,
        include_technicals=False,
        min_market_cap=8_000_000_000,
        max_pe=25.0,
        min_revenue_growth=10.0,
        include_latam=False,
    )

    df = payload["table"]
    notes = payload["notes"]
    source = payload["source"]

    table = df.set_index("ticker")
    assert set(table.index) == {"AAA", "BBB", "CCC"}
    assert table.loc["CCC"].notna().any()
    assert table.loc["AAA"].isna().all()
    assert table.loc["BBB"].isna().all()
    assert any("seleccionados automáticamente" in note for note in notes)
    assert any("min_market_cap" in note for note in notes)
    assert source == "yahoo"


def test_controller_reports_when_no_candidates(monkeypatch):
    client = AutoYahooClient({}, listings={"TEST": []})
    monkeypatch.setattr(ops, "YahooFinanceClient", lambda: client)
    monkeypatch.setattr(ops, "_get_target_markets", lambda: ["TEST"])

    payload = run_opportunities_controller(manual_tickers=None)

    df = payload["table"]
    notes = payload["notes"]
    source = payload["source"]

    assert df.empty
    assert any("No se encontraron símbolos" in note for note in notes)
    assert source == "yahoo"


def test_controller_uses_configured_markets(monkeypatch, auto_dataset):
    listings = {
        "US": [{"ticker": "AAA"}, {"ticker": "CCC"}],
        "LATAM": [{"ticker": "BBB"}],
    }

    client = AutoYahooClient(auto_dataset, listings=listings)
    monkeypatch.setattr(ops, "YahooFinanceClient", lambda: client)
    monkeypatch.setattr(ops, "_get_target_markets", lambda: ["LATAM"])

    payload = run_opportunities_controller(
        manual_tickers=None,
        include_technicals=False,
        include_latam=True,
    )

    df = payload["table"]
    notes = payload["notes"]
    source = payload["source"]

    assert list(df["ticker"]) == ["BBB"]
    assert source == "yahoo"
    assert any("seleccionados automáticamente" in note for note in notes)
