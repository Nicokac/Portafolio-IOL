from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from application.screener.opportunities import run_screener_stub


def _tickers(df: pd.DataFrame) -> set[str]:
    return set(df["ticker"].dropna().astype(str))


def test_filters_apply_to_base_dataset() -> None:
    df = run_screener_stub(
        min_market_cap=100_000,
        max_pe=35.0,
        min_revenue_growth=5.0,
        include_latam=False,
    )
    assert _tickers(df) == {"AAPL", "MSFT"}
    assert (df["market_cap"] >= 100_000).all()
    assert (df["pe_ratio"] <= 35.0).all()
    assert (df["revenue_growth"] >= 5.0).all()
    assert not df.get("is_latam", pd.Series(dtype=bool)).any()
    assert {"rsi", "sma_50", "sma_200"}.isdisjoint(df.columns)


def test_include_latam_flag_keeps_companies_when_enabled() -> None:
    df = run_screener_stub(include_latam=True, include_technicals=True)
    assert "MELI" in _tickers(df)
    assert bool(df.loc[df["ticker"] == "MELI", "is_latam"].iloc[0])
    assert {"rsi", "sma_50", "sma_200"}.issubset(df.columns)


def test_manual_tickers_return_placeholder_after_filters() -> None:
    df = run_screener_stub(
        manual_tickers=["MELI"],
        include_latam=False,
        max_pe=20.0,
    )
    assert "MELI" in _tickers(df)
    # MELI is excluded by filters, so its numeric fields should be NaN after placeholder padding.
    row = df.loc[df["ticker"] == "MELI"].iloc[0]
    assert pd.isna(row["pe_ratio"])
    assert pd.isna(row["market_cap"])
