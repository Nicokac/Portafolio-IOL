"""Stub implementation for the opportunities screener.

This module provides a small, deterministic dataset that emulates the
structure returned by a future screener implementation. The goal is to
allow UI components and controllers to be developed and tested without the
final data provider in place.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import pandas as pd

# Base dataset used to simulate screener output.
_BASE_OPPORTUNITIES = [
    {
        "ticker": "AAPL",
        "payout_ratio": 18.5,
        "dividend_streak": 12,
        "cagr": 14.2,
        "dividend_yield": 0.55,
        "price": 180.12,
        "rsi": 56.8,
        "sma_50": 172.34,
        "sma_200": 158.44,
    },
    {
        "ticker": "MSFT",
        "payout_ratio": 28.3,
        "dividend_streak": 20,
        "cagr": 11.7,
        "dividend_yield": 0.82,
        "price": 325.74,
        "rsi": 48.9,
        "sma_50": 312.66,
        "sma_200": 289.14,
    },
    {
        "ticker": "KO",
        "payout_ratio": 73.0,
        "dividend_streak": 61,
        "cagr": 7.5,
        "dividend_yield": 2.94,
        "price": 58.31,
        "rsi": 44.1,
        "sma_50": 59.0,
        "sma_200": 60.8,
    },
    {
        "ticker": "JNJ",
        "payout_ratio": 51.2,
        "dividend_streak": 59,
        "cagr": 6.9,
        "dividend_yield": 2.77,
        "price": 160.5,
        "rsi": 52.7,
        "sma_50": 158.9,
        "sma_200": 165.2,
    },
    {
        "ticker": "NUE",
        "payout_ratio": 33.4,
        "dividend_streak": 49,
        "cagr": 9.8,
        "dividend_yield": 1.37,
        "price": 165.8,
        "rsi": 62.1,
        "sma_50": 160.3,
        "sma_200": 155.6,
    },
]


def _normalise_tickers(manual_tickers: Optional[Sequence[str]]) -> List[str]:
    """Normalise tickers to uppercase strings without duplicates."""

    if not manual_tickers:
        return []
    if isinstance(manual_tickers, str):
        manual_tickers = [manual_tickers]
    seen = set()
    cleaned: List[str] = []
    for ticker in manual_tickers:
        if ticker is None:
            continue
        ticker_clean = str(ticker).strip().upper()
        if not ticker_clean or ticker_clean in seen:
            continue
        seen.add(ticker_clean)
        cleaned.append(ticker_clean)
    return cleaned


def _append_placeholder_rows(df: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Ensure that the DataFrame contains rows for each provided ticker."""

    if not tickers:
        return df
    df = df.set_index("ticker")
    df = df.reindex(tickers)
    df.index.name = "ticker"
    return df.reset_index()


def run_screener_stub(
    *,
    manual_tickers: Optional[Iterable[str]] = None,
    max_payout: Optional[float] = None,
    min_div_streak: Optional[int] = None,
    min_cagr: Optional[float] = None,
    include_technicals: bool = False,
) -> pd.DataFrame:
    """Return a filtered sample dataset that mimics a screener output.

    Parameters
    ----------
    manual_tickers:
        Optional iterable of tickers to focus on. When provided, the returned
        DataFrame will contain rows for these tickers (missing data will be
        represented with ``NaN`` values).
    max_payout:
        Maximum payout ratio percentage allowed. Rows exceeding this value are
        dropped.
    min_div_streak:
        Minimum number of consecutive years of dividend growth required.
    min_cagr:
        Minimum compound annual growth rate required.
    include_technicals:
        When ``True`` technical indicators (e.g., RSI, SMAs) are included in the
        DataFrame. When ``False`` those columns are omitted.
    """

    df = pd.DataFrame(_BASE_OPPORTUNITIES)

    if max_payout is not None:
        df = df[df["payout_ratio"] <= max_payout]
    if min_div_streak is not None:
        df = df[df["dividend_streak"] >= min_div_streak]
    if min_cagr is not None:
        df = df[df["cagr"] >= min_cagr]

    manual = _normalise_tickers(manual_tickers)
    if manual:
        df = df[df["ticker"].isin(manual)]
        df = _append_placeholder_rows(df, manual)

    df = df.copy()
    payout_component = (100 - df["payout_ratio"]).clip(lower=0, upper=100) * 0.4
    streak_component = df["dividend_streak"].clip(lower=0) * 0.3
    cagr_component = df["cagr"].clip(lower=0) * 0.3
    df["score_compuesto"] = (
        payout_component + streak_component + cagr_component
    ) / 10.0

    technical_columns = ["rsi", "sma_50", "sma_200"]
    if not include_technicals:
        df = df.drop(columns=[c for c in technical_columns if c in df.columns])

    df = df.reset_index(drop=True)
    return df


__all__ = ["run_screener_stub"]
