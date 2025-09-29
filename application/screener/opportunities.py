"""Stub implementation for the opportunities screener.

This module provides a small, deterministic dataset that emulates the
structure returned by a future screener implementation. The goal is to
allow UI components and controllers to be developed and tested without the
final data provider in place.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterable, List, Optional, Sequence

import pandas as pd

from infrastructure.market import YahooFinanceClient
from shared.errors import AppError

LOGGER = logging.getLogger(__name__)

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


def _is_valid_number(value: float | int | pd.NA | None) -> bool:
    return value is not None and not pd.isna(value)


def _safe_round(value: float | pd.NA | None, digits: int = 2) -> float | pd.NA:
    if not _is_valid_number(value):
        return pd.NA
    return round(float(value), digits)


def _compute_cagr(price_history: pd.DataFrame | None, years: int) -> float | pd.NA:
    if price_history is None or price_history.empty:
        return pd.NA
    data = price_history.sort_values("date")["date"].reset_index(drop=True)
    prices = price_history.sort_values("date")["adj_close"].astype(float).reset_index(drop=True)
    end_price = prices.iloc[-1]
    if end_price <= 0:
        return pd.NA
    cutoff = data.iloc[-1] - pd.DateOffset(years=years)
    mask = data <= cutoff
    if not mask.any():
        return pd.NA
    start_price = prices[mask].iloc[-1]
    if start_price <= 0:
        return pd.NA
    cagr = (end_price / start_price) ** (1 / years) - 1
    return float(cagr * 100.0)


def _compute_dividend_streak(dividends: pd.DataFrame | None) -> int | pd.NA:
    if dividends is None or dividends.empty:
        return pd.NA
    df = dividends.copy()
    df["year"] = df["date"].dt.year
    annual = df.groupby("year")["amount"].sum().sort_index()
    if annual.empty:
        return pd.NA
    streak = 0
    previous = None
    for _, value in reversed(list(annual.items())):
        if value <= 0:
            break
        if previous is None:
            streak = 1
            previous = value
            continue
        if value >= previous:
            streak += 1
            previous = value
        else:
            break
    return streak or pd.NA


def _compute_buyback_ratio(shares: pd.DataFrame | None) -> float | pd.NA:
    if shares is None or shares.empty:
        return pd.NA
    df = shares.sort_values("date")
    start = float(df["shares"].iloc[0])
    end = float(df["shares"].iloc[-1])
    if start <= 0 or end <= 0:
        return pd.NA
    return float((start - end) / start * 100.0)


def _compute_rsi(price_history: pd.DataFrame | None, period: int = 14) -> float | pd.NA:
    if price_history is None or len(price_history) <= period:
        return pd.NA
    closes = price_history.sort_values("date")["adj_close"].astype(float)
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    if avg_gain.empty or avg_loss.empty:
        return pd.NA
    last_gain = avg_gain.iloc[-1]
    last_loss = avg_loss.iloc[-1]
    if pd.isna(last_gain) or pd.isna(last_loss):
        return pd.NA
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return float(100 - (100 / (1 + rs)))


def _compute_macd(price_history: pd.DataFrame | None, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float | pd.NA, float | pd.NA]:
    if price_history is None or len(price_history) < slow + signal:
        return pd.NA, pd.NA
    closes = price_history.sort_values("date")["adj_close"].astype(float)
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    macd_value = macd_line.iloc[-1]
    hist_value = hist.iloc[-1]
    if pd.isna(macd_value) or pd.isna(hist_value):
        return pd.NA, pd.NA
    return float(macd_value), float(hist_value)


def _compute_sma(price_history: pd.DataFrame | None, window: int) -> float | pd.NA:
    if price_history is None or len(price_history) < window:
        return pd.NA
    closes = price_history.sort_values("date")["adj_close"].astype(float)
    sma = closes.rolling(window=window).mean().iloc[-1]
    return float(sma) if not pd.isna(sma) else pd.NA


def _compute_score(metrics: dict[str, float | int | pd.NA]) -> float | pd.NA:
    contributions: list[float] = []

    payout = metrics.get("payout_ratio")
    if _is_valid_number(payout):
        contributions.append(max(0.0, min(100.0, 100.0 - float(payout))) / 10.0)

    streak = metrics.get("dividend_streak")
    if _is_valid_number(streak):
        contributions.append(min(float(streak), 50.0) / 5.0)

    cagr = metrics.get("cagr")
    if _is_valid_number(cagr):
        contributions.append(max(0.0, min(25.0, float(cagr))) / 2.5)

    buyback = metrics.get("buyback")
    if _is_valid_number(buyback):
        contributions.append(max(0.0, min(20.0, float(buyback))) / 2.0)

    rsi = metrics.get("rsi")
    if _is_valid_number(rsi):
        contributions.append(max(0.0, 100.0 - abs(float(rsi) - 50.0) * 2.0) / 10.0)

    macd_hist = metrics.get("macd_hist")
    if _is_valid_number(macd_hist):
        contributions.append((max(-5.0, min(5.0, float(macd_hist))) + 5.0))

    if not contributions:
        return pd.NA

    score = sum(contributions) / len(contributions)
    return round(score, 2)


def _fetch_with_warning(
    fetcher: Callable[[str], object], ticker: str, label: str
) -> object | None:
    try:
        return fetcher(ticker)
    except AppError as exc:
        LOGGER.warning("Faltan datos de %s para %s: %s", label, ticker, exc)
        return None


def _output_columns(include_technicals: bool) -> list[str]:
    columns = [
        "ticker",
        "payout_ratio",
        "dividend_streak",
        "cagr",
        "dividend_yield",
        "price",
        "rsi",
        "sma_50",
        "sma_200",
        "score_compuesto",
    ]
    if include_technicals:
        return columns
    return [c for c in columns if c not in {"rsi", "sma_50", "sma_200"}]


_LATAM_COUNTRIES = {
    "argentina",
    "bolivia",
    "brazil",
    "chile",
    "colombia",
    "costa rica",
    "cuba",
    "dominican republic",
    "ecuador",
    "el salvador",
    "guatemala",
    "honduras",
    "mexico",
    "nicaragua",
    "panama",
    "paraguay",
    "peru",
    "uruguay",
    "venezuela",
}


def _as_optional_float(value: object) -> float | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    try:
        return float(value)
    except (TypeError, ValueError):
        return pd.NA


def _is_latam_country(country: object | None) -> bool:
    if not country:
        return False
    normalized = str(country).strip().lower()
    if not normalized:
        return False
    return normalized in _LATAM_COUNTRIES


def run_screener_yahoo(
    *,
    manual_tickers: Optional[Iterable[str]] = None,
    max_payout: Optional[float] = None,
    min_div_streak: Optional[int] = None,
    min_cagr: Optional[float] = None,
    include_technicals: bool = False,
    min_market_cap: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_revenue_growth: Optional[float] = None,
    include_latam: Optional[bool] = None,
    client: YahooFinanceClient | None = None,
) -> pd.DataFrame:
    """Run the Yahoo-based screener returning the same schema as the stub."""

    tickers = _normalise_tickers(manual_tickers)
    if not tickers:
        columns = _output_columns(include_technicals)
        return pd.DataFrame(columns=columns)

    client = client or YahooFinanceClient()
    rows: list[dict[str, object]] = []

    for ticker in tickers:
        fundamentals = _fetch_with_warning(client.get_fundamentals, ticker, "fundamentals")
        dividends = _fetch_with_warning(client.get_dividends, ticker, "dividendos")
        shares = _fetch_with_warning(client.get_shares_outstanding, ticker, "acciones")
        prices = _fetch_with_warning(client.get_price_history, ticker, "precios")

        payout = _safe_round(fundamentals["payout_ratio"] if fundamentals else pd.NA)
        dividend_yield = _safe_round(
            fundamentals["dividend_yield"] if fundamentals else pd.NA
        )
        dividend_streak = _compute_dividend_streak(dividends)

        cagr_values = []
        for years in (3, 5):
            value = _compute_cagr(prices, years)
            if _is_valid_number(value):
                cagr_values.append(float(value))
        cagr = _safe_round(sum(cagr_values) / len(cagr_values) if cagr_values else pd.NA)

        price = pd.NA
        if prices is not None and not prices.empty:
            price = _safe_round(prices.sort_values("date")["close"].iloc[-1])

        rsi = _safe_round(_compute_rsi(prices))
        sma_50 = _safe_round(_compute_sma(prices, 50))
        sma_200 = _safe_round(_compute_sma(prices, 200))
        _, macd_hist = _compute_macd(prices)
        macd_hist = _safe_round(macd_hist, digits=4) if _is_valid_number(macd_hist) else pd.NA
        buyback = _safe_round(_compute_buyback_ratio(shares))

        metrics = {
            "payout_ratio": payout,
            "dividend_streak": dividend_streak,
            "cagr": cagr,
            "buyback": buyback,
            "rsi": rsi,
            "macd_hist": macd_hist,
        }
        score = _compute_score(metrics)

        market_cap = _as_optional_float(
            fundamentals.get("market_cap") if fundamentals else pd.NA
        )
        pe_ratio = _as_optional_float(
            fundamentals.get("pe_ratio")
            if fundamentals and "pe_ratio" in fundamentals
            else (fundamentals.get("trailingPE") if fundamentals else pd.NA)
        )
        if pe_ratio is pd.NA and fundamentals and "pe" in fundamentals:
            pe_ratio = _as_optional_float(fundamentals.get("pe"))
        revenue_growth = _as_optional_float(
            fundamentals.get("revenue_growth") if fundamentals else pd.NA
        )
        is_latam = False
        if fundamentals:
            country = fundamentals.get("country") or fundamentals.get("region")
            is_latam = _is_latam_country(country)

        row = {
            "ticker": ticker,
            "payout_ratio": payout,
            "dividend_streak": dividend_streak,
            "cagr": cagr,
            "dividend_yield": dividend_yield,
            "price": price,
            "rsi": rsi,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "score_compuesto": score,
            "_meta_market_cap": market_cap,
            "_meta_pe_ratio": pe_ratio,
            "_meta_revenue_growth": revenue_growth,
            "_meta_is_latam": is_latam,
        }

        rows.append(row)

    df = pd.DataFrame(rows).convert_dtypes()

    if max_payout is not None:
        df = df[df["payout_ratio"].isna() | (df["payout_ratio"] <= max_payout)]
    if min_div_streak is not None:
        df = df[df["dividend_streak"].isna() | (df["dividend_streak"] >= min_div_streak)]
    if min_cagr is not None:
        df = df[df["cagr"].isna() | (df["cagr"] >= min_cagr)]
    if min_market_cap is not None:
        df = df[
            df["_meta_market_cap"].isna()
            | (df["_meta_market_cap"] >= float(min_market_cap))
        ]
    if max_pe is not None:
        df = df[
            df["_meta_pe_ratio"].isna() | (df["_meta_pe_ratio"] <= float(max_pe))
        ]
    if min_revenue_growth is not None:
        df = df[
            df["_meta_revenue_growth"].isna()
            | (df["_meta_revenue_growth"] >= float(min_revenue_growth))
        ]

    include_latam_flag = True if include_latam is None else bool(include_latam)
    if not include_latam_flag:
        df = df[~df["_meta_is_latam"].fillna(False)]

    df = _append_placeholder_rows(df, tickers)

    df = df.drop(
        columns=[
            c
            for c in ("_meta_market_cap", "_meta_pe_ratio", "_meta_revenue_growth", "_meta_is_latam")
            if c in df.columns
        ]
    )

    if not include_technicals:
        df = df[_output_columns(False)]
    else:
        df = df[_output_columns(True)]

    df = df.reset_index(drop=True)
    return df


__all__ = ["run_screener_stub", "run_screener_yahoo"]
