"""Regression guard ensuring stub and Yahoo runners produce consistent outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from application.screener import opportunities as ops
from tests.application.test_screener_yahoo import FakeYahooClient


@pytest.fixture
def synthetic_universe() -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    """Return a stub dataset and Yahoo payload built from the same inputs."""

    price_dates = pd.date_range("2008-01-01", periods=30, freq="6MS", tz="UTC")
    dividend_dates = pd.date_range("2014-01-01", periods=8, freq="YS", tz="UTC")

    stub_rows: list[dict[str, object]] = []
    yahoo_payload: dict[str, dict[str, object]] = {}

    configs = (
        {
            "ticker": "SYNC1",
            "base_price": 48.0,
            "growth": 0.06,
            "dividend_start": 1.2,
            "shares": 3_200_000,
            "dividend_yield": 2.4,
            "payout_ratio": 42.0,
            "market_cap": 180_000_000,
            "pe_ratio": 22.0,
            "revenue_growth": 8.5,
            "trailing_eps": 4.2,
            "forward_eps": 4.7,
            "sector": "Technology",
        },
        {
            "ticker": "SYNC2",
            "base_price": 52.0,
            "growth": 0.055,
            "dividend_start": 1.4,
            "shares": 2_800_000,
            "dividend_yield": 2.8,
            "payout_ratio": 38.0,
            "market_cap": 210_000_000,
            "pe_ratio": 20.0,
            "revenue_growth": 9.0,
            "trailing_eps": 4.6,
            "forward_eps": 5.2,
            "sector": "Healthcare",
        },
        {
            "ticker": "SYNC3",
            "base_price": 46.0,
            "growth": 0.058,
            "dividend_start": 1.1,
            "shares": 2_500_000,
            "dividend_yield": 2.1,
            "payout_ratio": 45.0,
            "market_cap": 195_000_000,
            "pe_ratio": 21.0,
            "revenue_growth": 7.5,
            "trailing_eps": 4.0,
            "forward_eps": 4.6,
            "sector": "Industrials",
        },
    )

    for entry in configs:
        ticker = entry["ticker"]
        growth_curve = np.power(1.0 + entry["growth"], np.arange(len(price_dates)))
        closes = entry["base_price"] * growth_curve
        prices = pd.DataFrame(
            {
                "date": price_dates,
                "close": closes,
                "adj_close": closes,
                "volume": np.linspace(10_000, 50_000, len(price_dates)),
            }
        )

        dividends = pd.DataFrame(
            {
                "date": dividend_dates,
                "amount": entry["dividend_start"] + np.linspace(0.0, 0.7, len(dividend_dates)),
            }
        )

        shares = pd.DataFrame(
            {
                "date": dividend_dates,
                "shares": entry["shares"] * np.power(0.985, np.arange(len(dividend_dates))),
            }
        )

        dividend_streak = ops._compute_dividend_streak(dividends)  # type: ignore[attr-defined]

        cagr_values: list[float] = []
        for years in (3, 5):
            value = ops._compute_cagr(prices, years)  # type: ignore[attr-defined]
            if ops._is_valid_number(value):  # type: ignore[attr-defined]
                cagr_values.append(float(value))
        cagr = ops._safe_round(sum(cagr_values) / len(cagr_values), digits=2) if cagr_values else pd.NA  # type: ignore[attr-defined]

        price_point = ops._safe_round(prices.sort_values("date")["close"].iloc[-1])  # type: ignore[attr-defined]
        rsi = ops._safe_round(ops._compute_rsi(prices))  # type: ignore[attr-defined]
        sma_50 = ops._safe_round(ops._compute_sma(prices, 50))  # type: ignore[attr-defined]
        sma_200 = ops._safe_round(ops._compute_sma(prices, 200))  # type: ignore[attr-defined]
        buyback = ops._safe_round(ops._compute_buyback_ratio(shares))  # type: ignore[attr-defined]

        fundamentals = {
            "ticker": ticker,
            "dividend_yield": entry["dividend_yield"],
            "payout_ratio": entry["payout_ratio"],
            "market_cap": entry["market_cap"],
            "pe_ratio": entry["pe_ratio"],
            "revenue_growth": entry["revenue_growth"],
            "country": "United States",
            "trailing_eps": entry["trailing_eps"],
            "forward_eps": entry["forward_eps"],
            "sector": entry["sector"],
        }

        stub_rows.append(
            {
                "ticker": ticker,
                "sector": entry["sector"],
                "payout_ratio": entry["payout_ratio"],
                "dividend_streak": dividend_streak,
                "cagr": cagr,
                "dividend_yield": entry["dividend_yield"],
                "price": price_point,
                "rsi": rsi,
                "sma_50": sma_50,
                "sma_200": sma_200,
                "market_cap": entry["market_cap"],
                "pe_ratio": entry["pe_ratio"],
                "revenue_growth": entry["revenue_growth"],
                "is_latam": False,
                "trailing_eps": entry["trailing_eps"],
                "forward_eps": entry["forward_eps"],
                "buyback": buyback,
            }
        )

        yahoo_payload[ticker] = {
            "fundamentals": fundamentals,
            "dividends": dividends,
            "shares": shares,
            "prices": prices,
        }

    return stub_rows, yahoo_payload


def _unwrap(result: pd.DataFrame | tuple[pd.DataFrame, list[str]]) -> tuple[pd.DataFrame, list[str]]:
    if isinstance(result, tuple):
        return result
    notes = list(result.attrs.get("_notes", []))
    return result, notes


def _strip_telemetry(notes: list[str]) -> list[str]:
    return [note for note in notes if "procesó" not in note]


@pytest.mark.parametrize(
    "filters",
    [
        {
            "max_payout": 55.0,
            "min_div_streak": 5,
            "min_cagr": 6.0,
            "min_market_cap": 150_000_000,
            "max_pe": 25.0,
            "min_revenue_growth": 7.0,
            "min_eps_growth": 8.0,
            "min_buyback": 1.0,
            "min_score_threshold": 5.0,
            "max_results": 5,
        }
    ],
)
def test_stub_and_yahoo_outputs_stay_aligned(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_universe: tuple[list[dict[str, object]], dict[str, dict[str, object]]],
    filters: dict[str, object],
) -> None:
    """Ensure that both screener runners honour filters in the same way."""

    stub_rows, yahoo_payload = synthetic_universe
    tickers = [row["ticker"] for row in stub_rows]

    monkeypatch.setattr(ops, "_BASE_OPPORTUNITIES", stub_rows, raising=False)

    stub_df, stub_notes = _unwrap(
        ops.run_screener_stub(manual_tickers=tickers, include_technicals=False, **filters)
    )
    yahoo_df, yahoo_notes = _unwrap(
        ops.run_screener_yahoo(
            manual_tickers=tickers,
            include_technicals=False,
            client=FakeYahooClient(yahoo_payload),
            **filters,
        )
    )

    stub_notes = _strip_telemetry(stub_notes)
    yahoo_notes = _strip_telemetry(yahoo_notes)

    differences: list[str] = []

    if list(stub_df["ticker"]) != list(yahoo_df["ticker"]):
        differences.append(
            "Tickers diverged: stub="
            + ", ".join(stub_df["ticker"].tolist())
            + "; yahoo="
            + ", ".join(yahoo_df["ticker"].tolist())
        )

    if stub_notes != yahoo_notes:
        differences.append(
            "Notes diverged: stub=" + str(stub_notes) + "; yahoo=" + str(yahoo_notes)
        )

    assert not differences, "Screener outputs diverged:\n" + "\n".join(differences)
