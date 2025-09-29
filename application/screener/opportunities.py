"""Stub implementation for the opportunities screener.

This module provides a small, deterministic dataset that emulates the
structure returned by a future screener implementation. The goal is to
allow UI components and controllers to be developed and tested without the
final data provider in place.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Callable, Iterable, List, Optional

import numpy as np
import pandas as pd

from infrastructure.market import YahooFinanceClient
from shared import config as shared_config
from shared.errors import AppError
from shared.settings import settings as shared_settings

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
        "market_cap": 2_800_000,
        "pe_ratio": 30.2,
        "revenue_growth": 7.4,
        "is_latam": False,
        "trailing_eps": 6.1,
        "forward_eps": 6.6,
        "buyback": 1.8,
        "sector": "Technology",

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
        "market_cap": 2_450_000,
        "pe_ratio": 33.5,
        "revenue_growth": 14.8,
        "is_latam": False,
        "trailing_eps": 9.2,
        "forward_eps": 9.8,
        "buyback": 1.1,
        "sector": "Technology",
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
        "market_cap": 260_000,
        "pe_ratio": 24.7,
        "revenue_growth": 4.3,
        "is_latam": False,
        "trailing_eps": 2.3,
        "forward_eps": 2.4,
        "buyback": 0.3,
        "sector": "Consumer Defensive",
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
        "market_cap": 415_000,
        "pe_ratio": 21.4,
        "revenue_growth": 3.1,
        "is_latam": False,
        "trailing_eps": 8.5,
        "forward_eps": 8.7,
        "buyback": 0.6,
        "sector": "Healthcare",
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
        "market_cap": 42_000,
        "pe_ratio": 8.9,
        "revenue_growth": -6.2,
        "is_latam": False,
        "trailing_eps": 18.4,
        "forward_eps": 18.6,
        "buyback": 0.0,
        "sector": "Basic Materials",
    },
    {
        "ticker": "MELI",
        "payout_ratio": 0.0,
        "dividend_streak": 0,
        "cagr": 0.0,
        "dividend_yield": 0.0,
        "price": 1225.5,
        "rsi": 58.2,
        "sma_50": 1187.4,
        "sma_200": 1042.8,
        "market_cap": 72_000,
        "pe_ratio": 76.4,
        "revenue_growth": 31.5,
        "is_latam": True,
        "trailing_eps": 4.8,
        "forward_eps": 6.2,
        "buyback": 0.0,
        "sector": "Consumer Cyclical",
    },
]

# Base universe used when the screener needs to propose tickers automatically.
# These values are intentionally static to guarantee deterministic behaviour in
# tests and during local development. The financial metrics are approximate and
# only serve to exercise the filtering logic when an external data provider is
# not available.
_DEFAULT_SYMBOL_POOL: list[dict[str, object]] = [
    {
        "ticker": "AAPL",
        "market_cap": 2_700_000_000_000,
        "pe": 29.4,
        "revenue_growth": 7.8,
        "region": "US",
    },
    {
        "ticker": "MSFT",
        "market_cap": 2_500_000_000_000,
        "pe": 33.7,
        "revenue_growth": 10.5,
        "region": "US",
    },
    {
        "ticker": "AMZN",
        "market_cap": 1_350_000_000_000,
        "pe": 60.1,
        "revenue_growth": 9.4,
        "region": "US",
    },
    {
        "ticker": "NVDA",
        "market_cap": 1_050_000_000_000,
        "pe": 45.3,
        "revenue_growth": 34.9,
        "region": "US",
    },
    {
        "ticker": "KO",
        "market_cap": 260_000_000_000,
        "pe": 24.1,
        "revenue_growth": 4.2,
        "region": "US",
    },
    {
        "ticker": "JNJ",
        "market_cap": 420_000_000_000,
        "pe": 15.9,
        "revenue_growth": 3.5,
        "region": "US",
    },
    {
        "ticker": "PG",
        "market_cap": 360_000_000_000,
        "pe": 24.6,
        "revenue_growth": 5.1,
        "region": "US",
    },
    {
        "ticker": "V",
        "market_cap": 510_000_000_000,
        "pe": 30.5,
        "revenue_growth": 12.3,
        "region": "US",
    },
    {
        "ticker": "MELI",
        "market_cap": 60_000_000_000,
        "pe": 78.0,
        "revenue_growth": 25.0,
        "region": "LATAM",
    },
    {
        "ticker": "GGAL",
        "market_cap": 7_500_000_000,
        "pe": 12.4,
        "revenue_growth": 18.2,
        "region": "LATAM",
    },
    {
        "ticker": "PAMP",
        "market_cap": 5_800_000_000,
        "pe": 10.7,
        "revenue_growth": 16.4,
        "region": "LATAM",
    },
    {
        "ticker": "VALE",
        "market_cap": 61_000_000_000,
        "pe": 6.8,
        "revenue_growth": 3.1,
        "region": "LATAM",
    },
]

_SYMBOL_POOL_ENV_VAR = "OPPORTUNITIES_SYMBOL_POOL"
_SYMBOL_POOL_FILE_ENV_VAR = "OPPORTUNITIES_SYMBOL_POOL_FILE"
_SYMBOL_POOL_CONFIG_KEY = "opportunities_symbol_pool"


def _normalise_symbol_pool(entries: object) -> list[dict[str, object]]:
    if not entries:
        return []

    normalised: list[dict[str, object]] = []
    seen: set[str] = set()

    if isinstance(entries, Mapping):
        for key, raw in entries.items():
            if isinstance(raw, Mapping):
                record = dict(raw)
                ticker = record.get("ticker") or key
            elif isinstance(raw, str):
                record = {}
                ticker = raw
            else:
                continue

            ticker_str = str(ticker or "").strip().upper()
            if not ticker_str or ticker_str in seen:
                continue

            record["ticker"] = ticker_str
            normalised.append(record)
            seen.add(ticker_str)
        return normalised

    for raw in entries:  # type: ignore[arg-type]
        if isinstance(raw, Mapping):
            record = dict(raw)
            ticker = record.get("ticker")
        elif isinstance(raw, str):
            record = {}
            ticker = raw
        else:
            continue

        ticker_str = str(ticker or "").strip().upper()
        if not ticker_str or ticker_str in seen:
            continue

        record["ticker"] = ticker_str
        normalised.append(record)
        seen.add(ticker_str)

    return normalised


def _load_symbol_pool_from_env() -> list[dict[str, object]] | None:
    file_path = os.getenv(_SYMBOL_POOL_FILE_ENV_VAR)
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("No se pudo leer %s: %s", file_path, exc)
        else:
            pool = _normalise_symbol_pool(data)
            if pool:
                return pool

    raw = os.getenv(_SYMBOL_POOL_ENV_VAR)
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            LOGGER.warning(
                "No se pudo interpretar OPPORTUNITIES_SYMBOL_POOL: %s", exc
            )
        else:
            pool = _normalise_symbol_pool(data)
            if pool:
                return pool

    return None


def _load_symbol_pool_from_config() -> list[dict[str, object]] | None:
    try:
        cfg = shared_config.get_config()
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("No se pudo cargar configuración: %s", exc)
        return None

    data = cfg.get(_SYMBOL_POOL_CONFIG_KEY)
    pool = _normalise_symbol_pool(data)
    return pool or None


def _get_symbol_pool() -> list[dict[str, object]]:
    pool = _load_symbol_pool_from_env()
    if pool is not None:
        return pool

    pool = _load_symbol_pool_from_config()
    if pool is not None:
        return pool

    return list(_DEFAULT_SYMBOL_POOL)


def _get_target_markets() -> list[str]:
    try:
        configured = getattr(shared_settings, "OPPORTUNITIES_TARGET_MARKETS", [])
    except Exception:  # pragma: no cover - defensive fallback
        configured = []

    if isinstance(configured, str):
        configured = [configured]

    markets: list[str] = []
    seen: set[str] = set()
    for entry in configured or []:
        market = str(entry or "").strip().upper()
        if not market or market in seen:
            continue
        markets.append(market)
        seen.add(market)

    if not markets:
        markets = ["NASDAQ", "NYSE", "AMEX"]

    return markets


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


def _apply_filters_and_finalize(
    df: pd.DataFrame,
    *,
    max_payout: Optional[float] = None,
    min_div_streak: Optional[int] = None,
    min_cagr: Optional[float] = None,
    min_market_cap: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_revenue_growth: Optional[float] = None,
    min_eps_growth: Optional[float] = None,
    min_buyback: Optional[float] = None,
    min_score_threshold: Optional[float] = None,
    max_results: Optional[int] = None,
    include_latam_flag: bool | None = True,
    include_technicals: bool,
    restrict_to_tickers: Sequence[str] | None = None,
    placeholder_tickers: Sequence[str] | None = None,
    exclude_tickers: Sequence[str] | None = None,
    market_cap_column: str = "market_cap",
    pe_ratio_column: str = "pe_ratio",
    revenue_growth_column: str = "revenue_growth",
    latam_column: str = "is_latam",
    trailing_eps_column: str = "trailing_eps",
    forward_eps_column: str = "forward_eps",
    buyback_column: str = "buyback",
    allowed_sectors: Sequence[str] | None = None,
    sector_column: str = "sector",
    allow_na_filters: bool = False,
    extra_drop_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Apply common filters and final adjustments for screener outputs."""

    result = df.copy()
    exclude_set: set[str] | None = None
    if exclude_tickers:
        exclude_set = {
            str(ticker).strip().upper()
            for ticker in exclude_tickers
            if str(ticker).strip()
        }
    normalized_allowed: set[str] | None = None
    if allowed_sectors:
        normalized_allowed = {str(value).casefold() for value in allowed_sectors}
    original_sector_series: pd.Series | None = None
    if sector_column in df.columns:
        original_sector_series = (
            df.set_index("ticker")[sector_column]
            .astype("string")
            .str.strip()
            .str.casefold()
        )

    if max_payout is not None and "payout_ratio" in result.columns:
        series = result["payout_ratio"]
        mask = series <= max_payout
        if allow_na_filters:
            mask = series.isna() | mask
        result = result[mask]

    if min_div_streak is not None and "dividend_streak" in result.columns:
        series = result["dividend_streak"]
        mask = series >= min_div_streak
        if allow_na_filters:
            mask = series.isna() | mask
        result = result[mask]

    if min_cagr is not None and "cagr" in result.columns:
        series = result["cagr"]
        mask = series >= min_cagr
        if allow_na_filters:
            mask = series.isna() | mask
        result = result[mask]

    if min_market_cap is not None and market_cap_column in result.columns:
        series = result[market_cap_column]
        mask = series >= float(min_market_cap)
        if allow_na_filters:
            mask = series.isna() | mask
        result = result[mask]

    if max_pe is not None and pe_ratio_column in result.columns:
        series = result[pe_ratio_column]
        mask = series <= float(max_pe)
        if allow_na_filters:
            mask = series.isna() | mask
        result = result[mask]

    if min_revenue_growth is not None and revenue_growth_column in result.columns:
        series = result[revenue_growth_column]
        mask = series >= float(min_revenue_growth)
        if allow_na_filters:
            mask = series.isna() | mask
        result = result[mask]

    if trailing_eps_column in result.columns:
        series = pd.to_numeric(result[trailing_eps_column], errors="coerce")
        mask = series > 0
        if allow_na_filters:
            mask = series.isna() | mask
        result = result[mask]

    if forward_eps_column in result.columns:
        series = pd.to_numeric(result[forward_eps_column], errors="coerce")
        mask = series > 0
        if allow_na_filters:
            mask = series.isna() | mask
        result = result[mask]

    if (
        min_eps_growth is not None
        and trailing_eps_column in result.columns
        and forward_eps_column in result.columns
    ):
        trailing = pd.to_numeric(result[trailing_eps_column], errors="coerce")
        forward = pd.to_numeric(result[forward_eps_column], errors="coerce")
        denominator = trailing.replace(0, pd.NA)
        growth = (forward - trailing) / denominator
        growth = growth.replace([np.inf, -np.inf], pd.NA) * 100.0
        mask = (trailing > 0) & (forward > 0) & (growth >= float(min_eps_growth))
        if allow_na_filters:
            mask = trailing.isna() | forward.isna() | growth.isna() | mask
        result = result[mask]

    if min_buyback is not None and buyback_column in result.columns:
        series = pd.to_numeric(result[buyback_column], errors="coerce")
        mask = series >= float(min_buyback)
        if allow_na_filters:
            mask = series.isna() | mask
        result = result[mask]

    if min_score_threshold is not None and "score_compuesto" in result.columns:
        series = pd.to_numeric(result["score_compuesto"], errors="coerce")
        mask = series >= float(min_score_threshold)
        if allow_na_filters:
            mask = series.isna() | mask
        result = result[mask]

    if include_latam_flag is False and latam_column in result.columns:
        result = result[~result[latam_column].fillna(False)]

    if normalized_allowed and sector_column in result.columns:
        result = result[
            result[sector_column]
            .astype("string")
            .str.strip()
            .str.casefold()
            .isin(normalized_allowed)
        ]

    if exclude_set and "ticker" in result.columns:
        normalized_tickers = result["ticker"].astype("string").str.upper()
        result = result[~normalized_tickers.isin(exclude_set)]

    if restrict_to_tickers:
        result = result[result["ticker"].isin(restrict_to_tickers)]

    if placeholder_tickers:
        placeholders = list(placeholder_tickers)
        if exclude_set:
            placeholders = [
                ticker
                for ticker in placeholders
                if str(ticker).strip().upper() not in exclude_set
            ]
        if normalized_allowed and original_sector_series is not None:
            placeholders = [
                ticker
                for ticker in placeholders
                if ticker in original_sector_series.index
                and original_sector_series[ticker] in normalized_allowed
            ]
        if placeholders:
            result = _append_placeholder_rows(result, placeholders)

    if extra_drop_columns:
        to_drop = [col for col in extra_drop_columns if col in result.columns]
        if to_drop:
            result = result.drop(columns=to_drop)

    if not include_technicals:
        technical_columns = ["rsi", "sma_50", "sma_200"]
        to_drop = [col for col in technical_columns if col in result.columns]
        if to_drop:
            result = result.drop(columns=to_drop)

    if max_results is not None:
        try:
            limit = int(max_results)
        except (TypeError, ValueError):
            limit = None
        else:
            if limit >= 0:
                result = result.head(limit)

    return result.reset_index(drop=True)


def run_screener_stub(
    *,
    manual_tickers: Optional[Iterable[str]] = None,
    exclude_tickers: Optional[Iterable[str]] = None,
    max_payout: Optional[float] = None,
    min_div_streak: Optional[int] = None,
    min_cagr: Optional[float] = None,
    min_market_cap: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_revenue_growth: Optional[float] = None,
    include_latam: bool = True,
    include_technicals: bool = False,
    min_eps_growth: Optional[float] = None,
    min_buyback: Optional[float] = None,
    sectors: Optional[Iterable[str]] = None,
    min_score_threshold: Optional[float] = None,
    max_results: Optional[int] = None,
) -> pd.DataFrame:
    """Return a filtered sample dataset that mimics a screener output.

    Parameters
    ----------
    manual_tickers:
        Optional iterable of tickers to focus on. When provided, the returned
        DataFrame will contain rows for these tickers (missing data will be
        represented with ``NaN`` values).
    exclude_tickers:
        Iterable opcional de símbolos que deben descartarse del resultado
        final incluso si aparecen en la base simulada o en la lista manual.
    max_payout:
        Maximum payout ratio percentage allowed. Rows exceeding this value are
        dropped.
    min_div_streak:
        Minimum number of consecutive years of dividend growth required.
    min_cagr:
        Minimum compound annual growth rate required.
    min_market_cap:
        Minimum market capitalisation (expressed in USD millions) required.
    max_pe:
        Maximum price to earnings ratio allowed.
    min_revenue_growth:
        Minimum year-over-year revenue growth percentage required.
    min_eps_growth:
        Minimum EPS growth percentage (forward vs trailing) required when
        available. When ``None`` the growth filter is skipped but EPS must be
        positive to remain in the dataset.
    min_buyback:
        Minimum buyback percentage (share reduction) required. ``None`` keeps
        companies regardless of their buyback ratio.
    min_score_threshold:
        Minimum composite score required. Rows below the threshold are
        discarded unless ``allow_na_filters`` is enabled.
    max_results:
        Maximum number of rows returned after applying all filters.
    include_latam:
        When ``False`` securities flagged as originating from Latin America are
        excluded from the results.
    include_technicals:
        When ``True`` technical indicators (e.g., RSI, SMAs) are included in the
        DataFrame. When ``False`` those columns are omitted.
    """

    df = pd.DataFrame(_BASE_OPPORTUNITIES)
    manual = _normalise_tickers(manual_tickers)
    excluded = set(_normalise_tickers(exclude_tickers))

    if excluded:
        df = df[~df["ticker"].isin(excluded)]
        manual = [ticker for ticker in manual if ticker not in excluded]

    df = df.copy()
    payout_component = (100 - df["payout_ratio"]).clip(lower=0, upper=100) * 0.4
    streak_component = df["dividend_streak"].clip(lower=0) * 0.3
    cagr_component = df["cagr"].clip(lower=0) * 0.3
    df["score_compuesto"] = (
        payout_component + streak_component + cagr_component
    ) / 10.0

    return _apply_filters_and_finalize(
        df,
        max_payout=max_payout,
        min_div_streak=min_div_streak,
        min_cagr=min_cagr,
        min_market_cap=min_market_cap,
        max_pe=max_pe,
        min_revenue_growth=min_revenue_growth,
        min_eps_growth=min_eps_growth,
        min_buyback=min_buyback,
        min_score_threshold=min_score_threshold,
        max_results=max_results,
        include_latam_flag=include_latam,
        include_technicals=include_technicals,
        restrict_to_tickers=manual or None,
        placeholder_tickers=manual or None,
        exclude_tickers=sorted(excluded) or None,
        market_cap_column="market_cap",
        pe_ratio_column="pe_ratio",
        revenue_growth_column="revenue_growth",
        latam_column="is_latam",
        extra_drop_columns=(
            "trailing_eps",
            "forward_eps",
            "buyback",
        ),

        allowed_sectors=_normalize_sector_filters(sectors),

    )


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
        "sector",
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


def _normalize_sector_name(value: object) -> str | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    return text.title()


def _normalize_sector_filters(values: Optional[Iterable[str]]) -> list[str]:
    if not values:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        normalized_value = _normalize_sector_name(raw)
        if normalized_value is pd.NA:
            continue
        key = str(normalized_value).casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(str(normalized_value))
    return normalized


def _as_optional_float(value: object) -> float | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    try:
        return float(value)
    except (TypeError, ValueError):
        return pd.NA


def _pick_first_numeric(values: Iterable[object]) -> float | pd.NA:
    for raw in values:
        result = _as_optional_float(raw)
        if result is not pd.NA:
            return result
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
    exclude_tickers: Optional[Iterable[str]] = None,
    max_payout: Optional[float] = None,
    min_div_streak: Optional[int] = None,
    min_cagr: Optional[float] = None,
    sectors: Optional[Iterable[str]] = None,
    include_technicals: bool = False,
    min_market_cap: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_revenue_growth: Optional[float] = None,
    include_latam: Optional[bool] = None,
    min_eps_growth: Optional[float] = None,
    min_buyback: Optional[float] = None,
    min_score_threshold: Optional[float] = None,
    max_results: Optional[int] = None,
    client: YahooFinanceClient | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    """Run the Yahoo-based screener returning the same schema as the stub.

    Parameters
    ----------
    exclude_tickers:
        Iterable opcional de símbolos a excluir de los resultados finales aun
        cuando aparezcan en la respuesta del proveedor de datos.
    """

    tickers = _normalise_tickers(manual_tickers)
    excluded = set(_normalise_tickers(exclude_tickers))
    if excluded:
        tickers = [ticker for ticker in tickers if ticker not in excluded]
    notes: list[str] = []
    using_default_universe = False
    listings_meta: dict[str, Mapping[str, object]] = {}
    target_markets: list[str] = []

    client = client or YahooFinanceClient()

    if not tickers:
        target_markets = _get_target_markets()
        listings = client.list_symbols_by_markets(target_markets)
        seen: set[str] = set()
        for entry in listings:
            if isinstance(entry, Mapping):
                metadata = dict(entry)
                ticker_value = metadata.get("ticker") or metadata.get("symbol")
            else:
                metadata = {}
                ticker_value = entry

            ticker_clean = str(ticker_value or "").strip().upper()
            if (
                not ticker_clean
                or ticker_clean in seen
                or ticker_clean in excluded
            ):
                continue

            metadata["ticker"] = ticker_clean
            listings_meta[ticker_clean] = metadata
            tickers.append(ticker_clean)
            seen.add(ticker_clean)

        using_default_universe = True

    if not tickers:
        columns = _output_columns(include_technicals)
        df = pd.DataFrame(columns=columns)
        if using_default_universe:
            message = "No se encontraron símbolos que cumplan los filtros especificados."
            if target_markets:
                message += " Mercados consultados: " + ", ".join(target_markets)
            notes.append(message)
        return (df, notes) if notes else df
    rows: list[dict[str, object]] = []
    sector_filters = _normalize_sector_filters(sectors)

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

        listing_meta = listings_meta.get(ticker, {})
        if not isinstance(listing_meta, Mapping):
            listing_meta = {}

        market_cap = _pick_first_numeric(
            (
                fundamentals.get("market_cap") if fundamentals else None,
                fundamentals.get("marketCap") if fundamentals else None,
                listing_meta.get("market_cap"),
                listing_meta.get("marketCap"),
            )
        )
        pe_ratio = _pick_first_numeric(
            (
                fundamentals.get("pe_ratio") if fundamentals else None,
                fundamentals.get("trailingPE") if fundamentals else None,
                fundamentals.get("pe") if fundamentals else None,
                listing_meta.get("pe_ratio"),
                listing_meta.get("trailingPE"),
                listing_meta.get("pe"),
            )
        )
        revenue_growth = _pick_first_numeric(
            (
                fundamentals.get("revenue_growth") if fundamentals else None,
                listing_meta.get("revenue_growth"),
            )
        )
        trailing_eps = _as_optional_float(
            fundamentals.get("trailing_eps") if fundamentals else pd.NA
        )
        if trailing_eps is pd.NA and fundamentals and "trailingEps" in fundamentals:
            trailing_eps = _as_optional_float(fundamentals.get("trailingEps"))
        forward_eps = _as_optional_float(
            fundamentals.get("forward_eps") if fundamentals else pd.NA
        )
        if forward_eps is pd.NA and fundamentals and "forwardEps" in fundamentals:
            forward_eps = _as_optional_float(fundamentals.get("forwardEps"))
        is_latam = False
        country: object | None = None
        sector = pd.NA
        if fundamentals:
            country = fundamentals.get("country") or fundamentals.get("region")
            sector = _normalize_sector_name(fundamentals.get("sector"))
        if not country and listing_meta:
            country = listing_meta.get("country") or listing_meta.get("region")
        is_latam = _is_latam_country(country)

        row = {
            "ticker": ticker,
            "sector": sector,
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
            "_meta_trailing_eps": trailing_eps,
            "_meta_forward_eps": forward_eps,
            "_meta_buyback": _as_optional_float(buyback),
        }

        rows.append(row)

    df = pd.DataFrame(rows).convert_dtypes()

    include_latam_flag = True if include_latam is None else bool(include_latam)

    df = _apply_filters_and_finalize(
        df,
        max_payout=max_payout,
        min_div_streak=min_div_streak,
        min_cagr=min_cagr,
        min_market_cap=min_market_cap,
        max_pe=max_pe,
        min_revenue_growth=min_revenue_growth,
        min_eps_growth=min_eps_growth,
        min_buyback=min_buyback,
        min_score_threshold=min_score_threshold,
        max_results=max_results,
        include_latam_flag=include_latam_flag,
        include_technicals=include_technicals,
        placeholder_tickers=tickers,
        exclude_tickers=sorted(excluded) or None,
        market_cap_column="_meta_market_cap",
        pe_ratio_column="_meta_pe_ratio",
        revenue_growth_column="_meta_revenue_growth",
        latam_column="_meta_is_latam",
        trailing_eps_column="_meta_trailing_eps",
        forward_eps_column="_meta_forward_eps",
        buyback_column="_meta_buyback",

        allowed_sectors=sector_filters,

        allow_na_filters=True,
        extra_drop_columns=(
            "_meta_market_cap",
            "_meta_pe_ratio",
            "_meta_revenue_growth",
            "_meta_is_latam",
            "_meta_trailing_eps",
            "_meta_forward_eps",
            "_meta_buyback",
        ),
    )

    if using_default_universe and tickers:
        filters_applied: list[str] = []
        if min_market_cap is not None:
            filters_applied.append(f"min_market_cap={min_market_cap}")
        if max_pe is not None:
            filters_applied.append(f"max_pe={max_pe}")
        if min_revenue_growth is not None:
            filters_applied.append(f"min_revenue_growth={min_revenue_growth}")
        if include_latam is not None:
            filters_applied.append(f"include_latam={bool(include_latam)}")

        message = f"📈 Analizando {len(tickers)} símbolos seleccionados automáticamente"
        if target_markets:
            message += " de " + ", ".join(target_markets)
        notes.append(message)
        if filters_applied:
            notes.append("Filtros aplicados: " + ", ".join(filters_applied))

    return (df, notes) if notes else df


__all__ = ["run_screener_stub", "run_screener_yahoo"]
