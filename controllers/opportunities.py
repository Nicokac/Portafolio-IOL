"""Controller helpers for the opportunities screener."""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from application.screener.opportunities import run_screener_stub

_EXPECTED_COLUMNS: Sequence[str] = (
    "ticker",
    "payout_ratio",
    "dividend_streak",
    "cagr",
    "dividend_yield",
    "price",
    "score_compuesto",
)
_TECHNICAL_COLUMNS: Sequence[str] = ("rsi", "sma_50", "sma_200")


def _clean_manual_tickers(manual_tickers: Optional[Iterable[str]]) -> List[str]:
    """Normalise manual tickers by stripping whitespace and uppercasing."""

    if not manual_tickers:
        return []
    if isinstance(manual_tickers, str):
        manual_tickers = [manual_tickers]
    cleaned: List[str] = []
    seen = set()
    for raw in manual_tickers:
        if raw is None:
            continue
        tickers = re.split(r"[\s,;]+", str(raw))
        for ticker in tickers:
            ticker_clean = ticker.strip().upper()
            if not ticker_clean or ticker_clean in seen:
                continue
            seen.add(ticker_clean)
            cleaned.append(ticker_clean)
    return cleaned


def _ensure_columns(df: pd.DataFrame, include_technicals: bool) -> pd.DataFrame:
    """Guarantee that the DataFrame exposes the expected schema."""

    df = df.copy()
    expected_columns = list(_EXPECTED_COLUMNS)
    if include_technicals:
        expected_columns.extend(col for col in _TECHNICAL_COLUMNS)

    for column in expected_columns:
        if column not in df.columns:
            df[column] = pd.NA

    # Drop unexpected technical columns when the flag is disabled.
    if not include_technicals:
        df = df.drop(columns=[c for c in _TECHNICAL_COLUMNS if c in df.columns])

    # Reorder columns for consistency.
    ordered_cols = [c for c in expected_columns if c in df.columns]
    return df[ordered_cols]


def run_opportunities_controller(
    *,
    manual_tickers: Optional[Iterable[str]] = None,
    max_payout: Optional[float] = None,
    min_div_streak: Optional[int] = None,
    min_cagr: Optional[float] = None,
    include_technicals: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run the opportunities screener and return the results and notes."""

    tickers = _clean_manual_tickers(manual_tickers)
    df = run_screener_stub(
        manual_tickers=tickers,
        max_payout=max_payout,
        min_div_streak=min_div_streak,
        min_cagr=min_cagr,
        include_technicals=include_technicals,
    )

    df = _ensure_columns(df, include_technicals)

    notes: List[str] = []
    if tickers:
        missing = []
        for ticker in tickers:
            rows = df[df["ticker"] == ticker]
            if rows.empty:
                missing.append(ticker)
            else:
                rows = rows.drop(columns=["ticker"], errors="ignore")
                if rows.isna().all(axis=None):
                    missing.append(ticker)
        if missing:
            notes.append(
                "No se encontraron datos para: " + ", ".join(sorted(set(missing)))
            )

    return df, notes


__all__ = ["run_opportunities_controller"]
