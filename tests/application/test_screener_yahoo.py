from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from application.screener import opportunities as ops
from controllers import opportunities as ctrl
from shared.errors import AppError


class FakeYahooClient:
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


class MissingYahooClient:
    def get_fundamentals(self, ticker: str) -> dict[str, object]:  # noqa: ARG002
        raise AppError("missing fundamentals")

    def get_dividends(self, ticker: str) -> pd.DataFrame:  # noqa: ARG002
        raise AppError("missing dividends")

    def get_shares_outstanding(self, ticker: str) -> pd.DataFrame:  # noqa: ARG002
        raise AppError("missing shares")

    def get_price_history(self, ticker: str) -> pd.DataFrame:  # noqa: ARG002
        raise AppError("missing prices")


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
    assert row["score_compuesto"] == ops._compute_score(metrics)


def test_run_screener_yahoo_marks_missing(caplog):
    client = MissingYahooClient()
    df = ops.run_screener_yahoo(manual_tickers=["zzz"], client=client, include_technicals=True)

    assert df.iloc[0]["ticker"] == "ZZZ"
    assert df.iloc[0]["score_compuesto"] is pd.NA
    assert any("faltan datos" in record.getMessage().lower() for record in caplog.records)


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
