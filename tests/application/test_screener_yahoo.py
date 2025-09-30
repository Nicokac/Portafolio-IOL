from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from application.screener import opportunities as ops
from controllers import opportunities as ctrl
from infrastructure.market import YahooFinanceClient
from shared.errors import AppError


class FakeYahooClient:
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


def build_bulk_fake_yahoo_client(
    count: int = 520,
) -> FakeYahooClient:
    """Return a fake Yahoo client seeded with hundreds of synthetic tickers."""

    price_dates = pd.date_range("2015-01-01", periods=84, freq="MS", tz="UTC")
    dividend_dates = pd.date_range("2014-01-01", periods=8, freq="YS", tz="UTC")
    share_dates = dividend_dates

    data: dict[str, dict[str, object]] = {}
    listings: dict[str, list[dict[str, object]]] = {"BULK": []}

    sectors = ("Technology", "Healthcare", "Industrials")

    for index in range(count):
        ticker = f"BULK{index:04d}"
        base_price = 45.0 + (index % 25)
        growth_rate = 0.05 + (index % 12) * 0.004
        price_trend = np.power(1.0 + growth_rate, np.arange(len(price_dates)))
        closes = base_price * price_trend

        dividends = 1.6 + (index % 5) * 0.05 - np.linspace(0.0, 0.6, len(dividend_dates))
        shares_start = 4_000_000 + index * 2_500
        shares_trend = shares_start * np.power(0.985, np.arange(len(share_dates)))

        sector = sectors[index % len(sectors)]
        payout_ratio = 35.0 + (index % 40)
        revenue_growth = 4.0 + (index % 7)
        market_cap = 450_000_000 + index * 4_500_000
        pe_ratio = 18.0 + (index % 12)
        trailing_eps = 2.5 + (index % 8) * 0.35
        forward_eps = trailing_eps * (1.12 + (index % 4) * 0.03)

        fundamentals = {
            "ticker": ticker,
            "dividend_yield": 2.0 + (index % 6) * 0.25,
            "payout_ratio": payout_ratio,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "revenue_growth": revenue_growth,
            "country": "United States",
            "trailing_eps": trailing_eps,
            "forward_eps": forward_eps,
            "sector": sector,
        }

        data[ticker] = {
            "fundamentals": fundamentals,
            "dividends": pd.DataFrame({
                "date": dividend_dates,
                "amount": dividends,
            }),
            "shares": pd.DataFrame({
                "date": share_dates,
                "shares": shares_trend,
            }),
            "prices": pd.DataFrame(
                {
                    "date": price_dates,
                    "close": closes,
                    "adj_close": closes,
                    "volume": np.linspace(1_000, 5_000, len(price_dates)),
                }
            ),
        }

        listings["BULK"].append(
            {
                "ticker": ticker,
                "market_cap": market_cap,
                "pe_ratio": pe_ratio,
                "revenue_growth": revenue_growth,
            }
        )

    return FakeYahooClient(data, listings=listings)


class MissingYahooClient:
    def get_fundamentals(self, ticker: str) -> dict[str, object]:  # noqa: ARG002
        raise AppError("missing fundamentals")

    def get_dividends(self, ticker: str) -> pd.DataFrame:  # noqa: ARG002
        raise AppError("missing dividends")

    def get_shares_outstanding(self, ticker: str) -> pd.DataFrame:  # noqa: ARG002
        raise AppError("missing shares")

    def get_price_history(self, ticker: str) -> pd.DataFrame:  # noqa: ARG002
        raise AppError("missing prices")

    def list_symbols_by_markets(self, markets: list[str]) -> list[dict[str, object]]:  # noqa: ARG002
        raise AppError("missing listings")


@pytest.fixture
def comprehensive_data() -> dict[str, dict[str, object]]:
    dates = pd.date_range("2015-01-01", periods=8 * 365, freq="D", tz="UTC")
    growth = 1.08 ** (np.arange(len(dates)) / 365)
    close = 100 * growth
    prices = pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "adj_close": close,
            "volume": np.linspace(1000, 2000, len(dates)),
        }
    )

    dividend_dates = pd.to_datetime(
        [
            "2018-03-01",
            "2019-03-01",
            "2020-03-01",
            "2021-03-01",
            "2022-03-01",
            "2023-03-01",
        ],
        utc=True,
    )
    dividends = pd.DataFrame({"date": dividend_dates, "amount": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]})

    shares_dates = pd.to_datetime(
        ["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"],
        utc=True,
    )
    shares = pd.DataFrame(
        {
            "date": shares_dates,
            "shares": [1_000_000, 990_000, 980_000, 970_000, 960_000, 950_000],
        }
    )

    fundamentals = {
        "ticker": "ABC",
        "dividend_yield": 2.0,
        "payout_ratio": 40.0,
        "market_cap": 1_200_000_000,
        "pe_ratio": 18.0,
        "revenue_growth": 7.5,
        "country": "United States",
        "trailing_eps": 5.2,
        "forward_eps": 5.9,

        "sector": "Technology",

    }

    return {
        "ABC": {
            "fundamentals": fundamentals,
            "dividends": dividends,
            "shares": shares,
            "prices": prices,
        }
    }


def test_run_screener_yahoo_computes_metrics(comprehensive_data):
    client = FakeYahooClient(comprehensive_data)
    df = ops.run_screener_yahoo(
        manual_tickers=["abc"], client=client, include_technicals=True
    )

    assert df.shape[0] == 1
    assert {"rsi", "sma_50", "sma_200"}.issubset(df.columns)
    row = df.iloc[0]

    prices = comprehensive_data["ABC"]["prices"]
    dividends = comprehensive_data["ABC"]["dividends"]
    shares = comprehensive_data["ABC"]["shares"]

    expected_cagrs = [
        ops._compute_cagr(prices, years) for years in (3, 5)
    ]
    expected_cagrs = [val for val in expected_cagrs if ops._is_valid_number(val)]
    expected_cagr = ops._safe_round(sum(expected_cagrs) / len(expected_cagrs))

    assert row["ticker"] == "ABC"
    assert row["sector"] == "Technology"
    assert row["payout_ratio"] == 40.0
    assert row["dividend_yield"] == 2.0
    assert row["dividend_streak"] == ops._compute_dividend_streak(dividends)
    assert row["cagr"] == expected_cagr
    assert row["rsi"] == ops._safe_round(ops._compute_rsi(prices))
    assert row["sma_50"] == ops._safe_round(ops._compute_sma(prices, 50))
    assert row["sma_200"] == ops._safe_round(ops._compute_sma(prices, 200))

    _, macd_hist = ops._compute_macd(prices)
    macd_hist = ops._safe_round(macd_hist, digits=4) if ops._is_valid_number(macd_hist) else pd.NA

    metrics = {
        "payout_ratio": row["payout_ratio"],
        "dividend_streak": row["dividend_streak"],
        "cagr": row["cagr"],
        "buyback": ops._safe_round(ops._compute_buyback_ratio(shares)),
        "rsi": row["rsi"],
        "macd_hist": macd_hist,
    }
    base_score = ops._compute_score(metrics)
    assert base_score is not pd.NA
    assert row["score_compuesto"] == pytest.approx(float(base_score), abs=1e-2)


def test_run_screener_yahoo_marks_missing(caplog):
    client = MissingYahooClient()
    result = ops.run_screener_yahoo(
        manual_tickers=["zzz"], client=client, include_technicals=True
    )

    assert isinstance(result, tuple)
    df, notes = result

    assert df.iloc[0]["ticker"] == "ZZZ"
    assert df.iloc[0]["score_compuesto"] is pd.NA
    assert "sector" in df.columns
    assert pd.isna(df.iloc[0]["sector"])
    assert any("EPS" in note for note in notes)
    assert any("faltan datos" in record.getMessage().lower() for record in caplog.records)


def test_run_screener_yahoo_discards_missing_eps(monkeypatch, comprehensive_data):
    base = comprehensive_data["ABC"]
    fundamentals_missing = base["fundamentals"].copy()
    fundamentals_missing.update({"ticker": "MISS"})
    fundamentals_missing.pop("trailing_eps", None)
    fundamentals_missing.pop("forward_eps", None)

    data = {
        "MISS": {
            "fundamentals": fundamentals_missing,
            "dividends": base["dividends"],
            "shares": base["shares"],
            "prices": base["prices"],
        }
    }

    listings = {"TEST": [{"ticker": "MISS"}]}
    client = FakeYahooClient(data, listings=listings)
    monkeypatch.setattr(ops, "_get_target_markets", lambda: ["TEST"])

    result = ops.run_screener_yahoo(manual_tickers=None, client=client, include_technicals=False)

    assert isinstance(result, tuple)
    df, notes = result

    assert df.empty
    assert any("EPS" in note for note in notes)
    combined = " ".join(notes)
    assert "MISS" in combined


def test_run_screener_yahoo_filters_and_optional_columns(comprehensive_data):
    other_prices = comprehensive_data["ABC"]["prices"].copy()
    fundamentals_bad = {
        "ticker": "BAD",
        "dividend_yield": 1.0,
        "payout_ratio": 90.0,
        "market_cap": 900_000_000,
        "pe_ratio": 30.0,
        "revenue_growth": -2.5,
        "country": "Canada",
        "trailing_eps": -1.0,
        "forward_eps": -0.5,

        "sector": "Industrials",

    }

    dividends_bad = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-01-01", "2019-01-01"], utc=True),
            "amount": [1.0, 0.9],
        }
    )
    shares_bad = comprehensive_data["ABC"]["shares"].copy()

    data = {
        "ABC": comprehensive_data["ABC"],
        "BAD": {
            "fundamentals": fundamentals_bad,
            "dividends": dividends_bad,
            "shares": shares_bad,
            "prices": other_prices,
        },
    }
    client = FakeYahooClient(data)

    df = ops.run_screener_yahoo(
        manual_tickers=["ABC", "BAD"],
        client=client,
        max_payout=50,
        min_div_streak=3,
        min_cagr=5,
        include_technicals=False,
    )

    assert list(df.columns) == [
        "ticker",
        "sector",
        "payout_ratio",
        "dividend_streak",
        "cagr",
        "dividend_yield",
        "price",
        "score_compuesto",
    ]
    assert not any(col.startswith("_meta") for col in df.columns)
    assert df.iloc[0]["ticker"] == "ABC"
    assert pd.isna(df.iloc[1]["payout_ratio"])

    df_filtered = ops.run_screener_yahoo(
        manual_tickers=["ABC", "BAD"],
        client=client,
        include_technicals=False,
        sectors=["technology"],
    )

    assert list(df_filtered["ticker"]) == ["ABC"]
    assert set(df_filtered["sector"].dropna()) == {"Technology"}


def test_run_screener_yahoo_auto_universe_drops_filtered(monkeypatch, comprehensive_data):
    base = comprehensive_data["ABC"]
    listings = {"TEST": [{"ticker": "ABC"}, {"ticker": "BAD"}]}

    bad_fundamentals = base["fundamentals"].copy()
    bad_fundamentals.update({"ticker": "BAD", "payout_ratio": 90.0})

    data = {
        "ABC": base,
        "BAD": {
            "fundamentals": bad_fundamentals,
            "dividends": base["dividends"],
            "shares": base["shares"],
            "prices": base["prices"],
        },
    }

    client = FakeYahooClient(data, listings=listings)
    monkeypatch.setattr(ops, "_get_target_markets", lambda: ["TEST"])

    result = ops.run_screener_yahoo(
        manual_tickers=None,
        client=client,
        include_technicals=False,
        max_payout=50.0,
    )

    if isinstance(result, tuple):
        df, _notes = result
    else:  # pragma: no cover - defensive guard
        df = result

    assert list(df["ticker"]) == ["ABC"]
    assert "BAD" not in set(df["ticker"])


def test_run_screener_yahoo_applies_extended_filters(comprehensive_data):
    latam_prices = comprehensive_data["ABC"]["prices"].copy()
    latam_fundamentals = {
        "ticker": "LAT",
        "dividend_yield": 3.0,
        "payout_ratio": 55.0,
        "market_cap": 2_500_000_000,
        "pe_ratio": 15.0,
        "revenue_growth": 6.0,
        "country": "Brazil",

        "trailing_eps": 3.4,
        "forward_eps": 3.6,

        "sector": "Financial Services",

    }
    latam_dividends = comprehensive_data["ABC"]["dividends"].copy()
    latam_shares = comprehensive_data["ABC"]["shares"].copy()

    small_fundamentals = {
        "ticker": "SMALL",
        "dividend_yield": 1.5,
        "payout_ratio": 20.0,
        "market_cap": 150_000_000,
        "pe_ratio": 35.0,
        "revenue_growth": 1.0,
        "country": "United States",
        "trailing_eps": 1.2,
        "forward_eps": 1.1,

        "sector": "Technology",

    }
    small_dividends = comprehensive_data["ABC"]["dividends"].copy()
    small_shares = comprehensive_data["ABC"]["shares"].copy()
    small_prices = comprehensive_data["ABC"]["prices"].copy()

    data = {
        "ABC": comprehensive_data["ABC"],
        "LAT": {
            "fundamentals": latam_fundamentals,
            "dividends": latam_dividends,
            "shares": latam_shares,
            "prices": latam_prices,
        },
        "SMALL": {
            "fundamentals": small_fundamentals,
            "dividends": small_dividends,
            "shares": small_shares,
            "prices": small_prices,
        },
    }

    client = FakeYahooClient(data)

    df = ops.run_screener_yahoo(
        manual_tickers=["ABC", "LAT", "SMALL"],
        client=client,
        include_technicals=False,
        min_market_cap=500_000_000,
        max_pe=20.0,
        min_revenue_growth=5.0,
        include_latam=False,
    )

    assert "sector" in df.columns
    assert list(df["ticker"]) == ["ABC", "LAT", "SMALL"]
    results = {row["ticker"]: row for _, row in df.iterrows()}
    assert results["ABC"]["payout_ratio"] == 40.0
    assert pd.isna(results["LAT"]["payout_ratio"])
    assert pd.isna(results["SMALL"]["payout_ratio"])

    df_latam = ops.run_screener_yahoo(
        manual_tickers=["LAT"],
        client=client,
        include_technicals=False,
        include_latam=True,
    )

    assert df_latam.shape[0] == 1
    assert df_latam.iloc[0]["ticker"] == "LAT"
    assert df_latam.iloc[0]["payout_ratio"] == 55.0


def test_run_screener_yahoo_filters_eps_and_buybacks(
    comprehensive_data: dict[str, dict[str, object]]
) -> None:
    base = comprehensive_data["ABC"]
    base_fundamentals = base["fundamentals"].copy()
    base_dividends = base["dividends"].copy()
    base_prices = base["prices"].copy()
    base_shares = base["shares"].copy()

    growth_fundamentals = base_fundamentals.copy()
    growth_fundamentals.update({"ticker": "GRO", "trailing_eps": 4.0, "forward_eps": 4.4})

    flat_fundamentals = base_fundamentals.copy()
    flat_fundamentals.update({"ticker": "FLT", "trailing_eps": 4.0, "forward_eps": 4.1})

    negative_fundamentals = base_fundamentals.copy()
    negative_fundamentals.update({"ticker": "NEG", "trailing_eps": -0.5, "forward_eps": 0.2})

    weak_buyback_fundamentals = base_fundamentals.copy()
    weak_buyback_fundamentals.update({"ticker": "BUY", "trailing_eps": 4.2, "forward_eps": 4.5})

    shares_negative = pd.DataFrame(
        {
            "date": base_shares["date"],
            "shares": [1_000_000, 1_010_000, 1_015_000, 1_020_000, 1_025_000, 1_030_000],
        }
    )

    data = {
        "GRO": {
            "fundamentals": growth_fundamentals,
            "dividends": base_dividends,
            "shares": base_shares,
            "prices": base_prices,
        },
        "FLT": {
            "fundamentals": flat_fundamentals,
            "dividends": base_dividends,
            "shares": base_shares,
            "prices": base_prices,
        },
        "NEG": {
            "fundamentals": negative_fundamentals,
            "dividends": base_dividends,
            "shares": base_shares,
            "prices": base_prices,
        },
        "BUY": {
            "fundamentals": weak_buyback_fundamentals,
            "dividends": base_dividends,
            "shares": shares_negative,
            "prices": base_prices,
        },
    }

    client = FakeYahooClient(data)

    df = ops.run_screener_yahoo(
        manual_tickers=["GRO", "FLT", "NEG", "BUY"],
        client=client,
        include_technicals=False,
        min_eps_growth=5.0,
        min_buyback=0.1,
    )

    assert list(df["ticker"]) == ["GRO", "FLT", "NEG", "BUY"]
    results = {row["ticker"]: row for _, row in df.iterrows()}

    assert not pd.isna(results["GRO"]["payout_ratio"])
    assert pd.isna(results["FLT"]["payout_ratio"])
    assert pd.isna(results["NEG"]["payout_ratio"])
    assert pd.isna(results["BUY"]["payout_ratio"])


def test_run_screener_yahoo_strict_growth_buyback_sector_and_score_filters(
    comprehensive_data: dict[str, dict[str, object]]
) -> None:
    base = comprehensive_data["ABC"]
    base_prices = base["prices"].copy()
    base_dividends = base["dividends"].copy()
    base_shares = base["shares"].copy()

    def make_shares(values: list[int]) -> pd.DataFrame:
        return pd.DataFrame({"date": base_shares["date"], "shares": values})

    def make_entry(
        ticker: str,
        *,
        sector: str,
        trailing_eps: float,
        forward_eps: float,
        shares: pd.DataFrame,
        payout_ratio: float | None = None,
        dividends: pd.DataFrame | None = None,
        prices: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        fundamentals = base["fundamentals"].copy()
        fundamentals.update(
            {
                "ticker": ticker,
                "sector": sector,
                "trailing_eps": trailing_eps,
                "forward_eps": forward_eps,
            }
        )
        if payout_ratio is not None:
            fundamentals["payout_ratio"] = payout_ratio
        dividend_data = dividends.copy() if dividends is not None else base_dividends.copy()
        price_data = prices.copy() if prices is not None else base_prices.copy()
        return {
            "fundamentals": fundamentals,
            "dividends": dividend_data,
            "shares": shares,
            "prices": price_data,
        }

    elite_shares = make_shares([1_000_000, 960_000, 920_000, 880_000, 840_000, 800_000])
    medic_shares = make_shares([1_000_000, 955_000, 910_000, 870_000, 835_000, 800_000])
    weak_buyback_shares = make_shares([1_000_000, 995_000, 990_000, 985_000, 980_000, 975_000])
    strong_dividends = pd.DataFrame(
        {
            "date": base_dividends["date"],
            "amount": [1.5, 1.5, 1.4, 1.4, 1.3, 1.3],
        }
    )
    momentum_prices = base_prices.copy()
    growth_curve = 1.15 ** (np.arange(len(momentum_prices)) / 365)
    momentum_prices["close"] = 100.0 * growth_curve
    momentum_prices["adj_close"] = 100.0 * growth_curve
    consistent_prices = base_prices.copy()
    consistent_prices["close"] = 100.0
    consistent_prices["adj_close"] = 100.0

    data = {
        "ELITE": make_entry(
            "ELITE",
            sector="Technology",
            trailing_eps=4.0,
            forward_eps=5.2,
            shares=elite_shares,
            payout_ratio=32.0,
            dividends=strong_dividends,
            prices=momentum_prices,
        ),
        "MEDIC": make_entry(
            "MEDIC",
            sector="Healthcare",
            trailing_eps=3.8,
            forward_eps=4.9,
            shares=medic_shares,
            payout_ratio=30.0,
            dividends=strong_dividends,
            prices=momentum_prices,
        ),
        "SLOWBUY": make_entry(
            "SLOWBUY",
            sector="Technology",
            trailing_eps=4.1,
            forward_eps=5.3,
            shares=weak_buyback_shares,
            dividends=strong_dividends,
            prices=momentum_prices,
        ),
        "OFFSECTOR": make_entry(
            "OFFSECTOR",
            sector="Utilities",
            trailing_eps=4.0,
            forward_eps=5.1,
            shares=elite_shares,
            dividends=strong_dividends,
            prices=momentum_prices,
        ),
        "LOWSCORE": make_entry(
            "LOWSCORE",
            sector="Healthcare",
            trailing_eps=3.9,
            forward_eps=4.8,
            shares=elite_shares,
            payout_ratio=78.0,
            dividends=strong_dividends,
            prices=consistent_prices,
        ),
    }

    listings = {"NASDAQ": [{"ticker": symbol} for symbol in data]}
    client = FakeYahooClient(data, listings=listings)

    result = ops.run_screener_yahoo(
        client=client,
        include_technicals=False,
        min_eps_growth=20.0,
        min_buyback=10.0,
        sectors=["technology", "healthcare"],
        min_score_threshold=48.0,
    )

    if isinstance(result, tuple):
        df, notes = result
    else:  # pragma: no cover - defensive guard
        df, notes = result, []

    assert set(df["ticker"]) == {"ELITE", "MEDIC"}
    assert {"SLOWBUY", "OFFSECTOR", "LOWSCORE"}.isdisjoint(set(df["ticker"]))
    assert all("Ningún ticker" not in note for note in notes)
    assert any("Analizando" in note for note in notes)
    assert pd.to_numeric(df["score_compuesto"], errors="coerce").min() >= 48.0
def test_run_screener_yahoo_uses_market_listings(monkeypatch, comprehensive_data):
    listings = {"TEST": [{"ticker": "ABC", "market_cap": 1_200_000_000}]}
    client = FakeYahooClient(comprehensive_data, listings=listings)
    monkeypatch.setattr(ops, "_get_target_markets", lambda: ["TEST"])

    result = ops.run_screener_yahoo(manual_tickers=None, client=client)

    if isinstance(result, tuple):
        df, notes = result
    else:  # pragma: no cover - defensive guard
        df, notes = result, []

    assert list(df["ticker"]) == ["ABC"]
    assert any("seleccionados automáticamente" in note for note in notes)
    assert any("TEST" in note for note in notes)


def test_run_screener_yahoo_truncates_large_universe(monkeypatch):
    client = build_bulk_fake_yahoo_client()
    listings = client.list_symbols_by_markets(["BULK"])
    assert len(listings) >= 500

    monkeypatch.setattr(ops, "_get_target_markets", lambda: ["BULK"])
    monkeypatch.setattr(
        ops.shared_settings,
        "OPPORTUNITIES_MIN_RESULTS",
        25,
        raising=False,
    )

    start = time.perf_counter()
    result = ops.run_screener_yahoo(
        manual_tickers=None,
        client=client,
        include_technicals=False,
        max_results=10,
        max_payout=60.0,
        min_div_streak=4,
        min_cagr=4.0,
        min_market_cap=200_000_000,
        max_pe=25.0,
        min_revenue_growth=3.0,
        min_eps_growth=5.0,
        min_buyback=1.0,
        sectors=["Technology", "Healthcare"],
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 4.0, f"Execution took too long: {elapsed:.2f} seconds"

    assert isinstance(result, tuple)
    df, notes = result

    assert len(df) == 10
    assert df["ticker"].str.startswith("BULK").all()
    scores = pd.to_numeric(df["score_compuesto"], errors="coerce").dropna()
    assert list(scores) == sorted(scores, reverse=True)

    assert any("máximo solicitado" in note for note in notes)
    assert any("Solo se encontraron" in note for note in notes)
    assert any("Analizando" in note for note in notes)
    assert any("Filtros aplicados" in note for note in notes)


def test_apply_filters_clamps_scores_and_limits_results(monkeypatch):
    base = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Tech", "Tech"],
            "payout_ratio": [20.0, 10.0, 15.0],
            "dividend_streak": [5, 12, 8],
            "cagr": [6.0, 12.0, 9.0],
            "dividend_yield": [1.0, 1.2, 1.1],
            "price": [100.0, 110.0, 105.0],
            "score_compuesto": [10.0, 20.0, 15.0],
            "trailing_eps": [5.0, 6.0, 5.5],
            "forward_eps": [5.5, 6.5, 6.0],
            "buyback": [0.5, 1.0, 0.8],
        }
    )
    monkeypatch.setattr(ops.shared_settings, "OPPORTUNITIES_MIN_RESULTS", 4, raising=False)

    result = ops._apply_filters_and_finalize(
        base,
        include_technicals=False,
        allow_na_filters=True,
        min_score_threshold=None,
        max_results=2,
    )

    notes = result.attrs.get("_notes", [])

    assert list(result["ticker"]) == ["BBB", "CCC"]
    assert list(result["score_compuesto"]) == pytest.approx([20.0, 15.0])
    assert any("máximo solicitado" in note for note in notes)
    assert any("mínimo esperado: 4" in note for note in notes)


def test_apply_filters_keeps_scores_equal_to_threshold():
    base = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Tech", "Tech"],
            "payout_ratio": [20.0, 30.0, 10.0],
            "dividend_streak": [5, 7, 9],
            "cagr": [6.0, 8.0, 12.0],
            "dividend_yield": [1.0, 1.1, 1.2],
            "price": [100.0, 110.0, 120.0],
            "score_compuesto": [50.0, 55.0, 70.0],
            "trailing_eps": [5.0, 5.0, 5.0],
            "forward_eps": [5.5, 5.5, 5.5],
            "buyback": [0.5, 0.5, 0.5],
        }
    )

    baseline = ops._apply_filters_and_finalize(
        base,
        include_technicals=False,
        allow_na_filters=False,
        min_score_threshold=None,
    )

    scores = baseline.set_index("ticker")["score_compuesto"].astype(float)
    threshold_value = float(scores["BBB"])
    below_value = float(scores["AAA"])
    assert threshold_value == pytest.approx(55.0)
    assert below_value == pytest.approx(50.0)
    assert below_value < threshold_value

    result = ops._apply_filters_and_finalize(
        base,
        include_technicals=False,
        allow_na_filters=False,
        min_score_threshold=threshold_value,
    )

    assert set(result["ticker"]) == {"BBB", "CCC"}
    threshold_row = result[result["ticker"] == "BBB"]
    assert not threshold_row.empty
    assert threshold_row["score_compuesto"].iloc[0] == pytest.approx(threshold_value)
    notes = result.attrs.get("_notes", [])
    assert not any("puntaje mínimo" in note for note in notes)


def test_run_screener_yahoo_applies_score_threshold_inclusively(
    comprehensive_data: dict[str, dict[str, object]]
) -> None:
    base = comprehensive_data["ABC"]

    weak_fundamentals = base["fundamentals"].copy()
    weak_fundamentals.update(
        {
            "ticker": "WEAK",
            "dividend_yield": 0.8,
            "payout_ratio": 85.0,
            "market_cap": 900_000_000,
            "pe_ratio": 28.0,
            "revenue_growth": 1.5,
            "trailing_eps": 4.0,
            "forward_eps": 3.8,
        }
    )

    weak_dividends = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-03-01",
                    "2021-03-01",
                    "2022-03-01",
                    "2023-03-01",
                ],
                utc=True,
            ),
            "amount": [1.0, 0.95, 0.92, 0.9],
        }
    )

    weak_shares = pd.DataFrame(
        {
            "date": base["shares"]["date"],
            "shares": [
                1_000_000,
                1_005_000,
                1_010_000,
                1_020_000,
                1_030_000,
                1_040_000,
            ],
        }
    )

    weak_dates = base["prices"]["date"].reset_index(drop=True)
    weak_trend = np.linspace(120.0, 80.0, len(weak_dates))
    weak_prices = pd.DataFrame(
        {
            "date": weak_dates,
            "close": weak_trend,
            "adj_close": weak_trend,
            "volume": np.linspace(1500, 500, len(weak_dates)),
        }
    )

    data = {
        "ABC": base,
        "WEAK": {
            "fundamentals": weak_fundamentals,
            "dividends": weak_dividends,
            "shares": weak_shares,
            "prices": weak_prices,
        },
    }

    client = FakeYahooClient(data)

    baseline = ops.run_screener_yahoo(
        manual_tickers=["ABC", "WEAK"],
        client=client,
        include_technicals=False,
    )

    if isinstance(baseline, tuple):
        baseline_df = baseline[0]
    else:
        baseline_df = baseline

    scores = baseline_df.set_index("ticker")["score_compuesto"].astype(float)
    threshold_value = float(scores["ABC"])
    below_value = float(scores["WEAK"])
    assert threshold_value == pytest.approx(28.17, abs=1e-2)
    assert below_value == pytest.approx(12.17, abs=1e-2)
    assert below_value < threshold_value

    filtered = ops.run_screener_yahoo(
        manual_tickers=["ABC", "WEAK"],
        client=client,
        include_technicals=False,
        min_score_threshold=threshold_value,
    )

    if isinstance(filtered, tuple):
        filtered_df = filtered[0]
    else:
        filtered_df = filtered

    assert "ABC" in set(filtered_df["ticker"])
    weak_row = filtered_df[filtered_df["ticker"] == "WEAK"]
    assert not weak_row.empty
    assert weak_row["score_compuesto"].isna().all()


def test_run_screener_yahoo_emits_note_when_score_threshold_excludes_all(
    comprehensive_data: dict[str, dict[str, object]]
) -> None:
    client = FakeYahooClient(comprehensive_data)

    result = ops.run_screener_yahoo(
        manual_tickers=["ABC"],
        client=client,
        min_score_threshold=150.0,
    )

    assert isinstance(result, tuple)
    df, notes = result
    assert list(df["ticker"]) == ["ABC"]
    assert df["score_compuesto"].isna().all()
    assert any("puntaje mínimo" in note for note in notes)


def test_run_opportunities_controller_calls_yahoo(monkeypatch, comprehensive_data):
    latam_free_data = {
        "ABC": comprehensive_data["ABC"],
    }

    fake_client = FakeYahooClient(latam_free_data)

    monkeypatch.setattr(ops, "YahooFinanceClient", lambda: fake_client)

    def _stub_not_expected(**_kwargs):
        raise AssertionError("stub should not be used when Yahoo succeeds")

    monkeypatch.setattr(ctrl, "run_screener_stub", _stub_not_expected)

    df, notes, source = ctrl.run_opportunities_controller(
        manual_tickers=["abc"],
        include_technicals=False,
        min_market_cap=500_000_000,
        max_pe=25.0,
        min_revenue_growth=5.0,
        include_latam=True,
    )

    assert not any("Datos simulados" in note for note in notes)
    assert {ticker for ticker in df["ticker"]} == {"ABC"}
    assert source == "yahoo"
    assert {"rsi", "sma_50", "sma_200"}.isdisjoint(df.columns)


def test_run_opportunities_controller_exposes_technicals(monkeypatch, comprehensive_data):
    fake_client = FakeYahooClient(comprehensive_data)

    monkeypatch.setattr(ops, "YahooFinanceClient", lambda: fake_client)
    monkeypatch.setattr(
        ctrl,
        "run_screener_stub",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("fallback should not run")),
    )

    df, notes, source = ctrl.run_opportunities_controller(
        manual_tickers=["abc"],
        include_technicals=True,
    )

    assert not notes
    assert source == "yahoo"
    assert {"rsi", "sma_50", "sma_200"}.issubset(df.columns)


def test_run_opportunities_controller_applies_new_filters(
    monkeypatch: pytest.MonkeyPatch, comprehensive_data: dict[str, dict[str, object]]
) -> None:
    base_fundamentals = comprehensive_data["ABC"]["fundamentals"].copy()
    base_shares = comprehensive_data["ABC"]["shares"].copy()
    base_prices = comprehensive_data["ABC"]["prices"].copy()

    abc_dividends = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-03-01",
                    "2021-03-01",
                    "2022-03-01",
                    "2023-03-01",
                ],
                utc=True,
            ),
            "amount": [1.5, 1.5, 1.5, 1.5],
        }
    )

    payout_fundamentals = base_fundamentals.copy()
    payout_fundamentals.update({"ticker": "PAY", "payout_ratio": 80.0})

    streak_dividends = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-03-01", "2023-03-01"], utc=True),
            "amount": [1.0, 0.4],
        }
    )
    streak_fundamentals = base_fundamentals.copy()
    streak_fundamentals.update({"ticker": "STK"})

    cagr_prices = base_prices.copy()
    cagr_prices["close"] = 100.0
    cagr_prices["adj_close"] = 100.0
    cagr_fundamentals = base_fundamentals.copy()
    cagr_fundamentals.update({"ticker": "CGR"})

    data = {
        "ABC": {
            "fundamentals": base_fundamentals.copy(),
            "dividends": abc_dividends,
            "shares": base_shares.copy(),
            "prices": base_prices.copy(),
        },
        "PAY": {
            "fundamentals": payout_fundamentals,
            "dividends": abc_dividends.copy(),
            "shares": base_shares.copy(),
            "prices": base_prices.copy(),
        },
        "STK": {
            "fundamentals": streak_fundamentals,
            "dividends": streak_dividends,
            "shares": base_shares.copy(),
            "prices": base_prices.copy(),
        },
        "CGR": {
            "fundamentals": cagr_fundamentals,
            "dividends": abc_dividends.copy(),
            "shares": base_shares.copy(),
            "prices": cagr_prices,
        },
    }

    client = FakeYahooClient(data)

    monkeypatch.setattr(ops, "YahooFinanceClient", lambda: client)

    def _stub_not_expected(**_kwargs):
        raise AssertionError("stub should not be used when Yahoo succeeds")

    monkeypatch.setattr(ctrl, "run_screener_stub", _stub_not_expected)

    df, notes, source = ctrl.run_opportunities_controller(
        manual_tickers=["abc", "pay", "stk", "cgr"],
        max_payout=50.0,
        min_div_streak=3,
        min_cagr=5.0,
        include_technicals=False,
    )

    assert list(df["ticker"]) == ["ABC", "PAY", "STK", "CGR"]
    results = {row["ticker"]: row for _, row in df.iterrows()}

    assert not pd.isna(results["ABC"]["payout_ratio"])
    assert float(results["ABC"]["payout_ratio"]) == pytest.approx(40.0)
    assert pd.isna(results["PAY"]["payout_ratio"])
    assert pd.isna(results["STK"]["dividend_streak"])
    assert pd.isna(results["CGR"]["cagr"])

    assert notes == ["No se encontraron datos para: CGR, PAY, STK"]
    assert source == "yahoo"


_STUB_TICKERS = {"AAPL", "MSFT", "KO", "JNJ", "NUE", "MELI"}


# Instrucciones completas en README.md#pruebas para habilitar la marca live_yahoo.
@pytest.mark.live_yahoo
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_YF") != "1",
    reason="Set RUN_LIVE_YF=1 to enable live Yahoo Finance checks.",
)
def test_run_screener_yahoo_live_integration_returns_real_symbols() -> None:
    """Prueba contra datos reales de Yahoo Finance (habilitar con RUN_LIVE_YF=1)."""

    client = YahooFinanceClient()
    result = ops.run_screener_yahoo(
        client=client,
        include_technicals=False,
        min_market_cap=5_000_000_000,
        max_pe=40.0,
        min_revenue_growth=0.0,
        max_results=30,
    )

    df = result[0] if isinstance(result, tuple) else result

    assert not df.empty, "Expected live screener to return at least one ticker"

    tickers = {str(ticker).strip().upper() for ticker in df["ticker"].dropna()}
    non_stub_tickers = {ticker for ticker in tickers if ticker not in _STUB_TICKERS}

    assert len(non_stub_tickers) >= 5, (
        "Expected at least five tickers absent from the fictitious portfolio; "
        f"got {sorted(non_stub_tickers)} from {sorted(tickers)}"
    )
