from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from application.screener.opportunities import run_screener_stub
import application.screener.opportunities as opportunities_module


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


def test_run_screener_stub_applies_score_threshold_inclusively() -> None:
    baseline = run_screener_stub(include_technicals=False)
    ordered = baseline.sort_values("score_compuesto", ascending=True).reset_index(drop=True)

    below_row = ordered.iloc[0]
    threshold_row = ordered.iloc[1]
    threshold_value = float(threshold_row["score_compuesto"])

    assert float(below_row["score_compuesto"]) == pytest.approx(36.72, abs=1e-2)
    assert threshold_value == pytest.approx(49.34, abs=1e-2)

    assert threshold_value > float(below_row["score_compuesto"])

    filtered = run_screener_stub(
        include_technicals=False,
        min_score_threshold=threshold_value,
    )

    assert threshold_row["ticker"] in _tickers(filtered)
    assert below_row["ticker"] not in _tickers(filtered)


@pytest.mark.timeout(2)
def test_run_screener_stub_truncates_large_universe(monkeypatch: pytest.MonkeyPatch) -> None:
    synthetic_universe: list[dict[str, object]] = []
    for index in range(240):
        tier = index % 3
        ticker = f"SYN{index:03d}"
        synthetic_universe.append(
            {
                "ticker": ticker,
                "payout_ratio": 20.0 if tier == 0 else 40.0 if tier == 1 else 70.0,
                "dividend_streak": 30 if tier == 0 else 15 if tier == 1 else 5,
                "cagr": 20.0 if tier == 0 else 10.0 if tier == 1 else 5.0,
                "dividend_yield": 1.5 + tier,
                "price": 100.0 + index,
                "rsi": 50.0 if tier == 0 else 60.0 if tier == 1 else 80.0,
                "sma_50": 95.0 + index,
                "sma_200": 90.0 + index,
                "market_cap": 1_000 + index * 5,
                "pe_ratio": 15.0 if tier == 0 else 25.0 if tier == 1 else 35.0,
                "revenue_growth": 15.0 if tier == 0 else 8.0 if tier == 1 else -2.0,
                "is_latam": bool(index % 10 == 0),
                "trailing_eps": 5.0,
                "forward_eps": 5.5,
                "buyback": 10.0 if tier == 0 else 5.0 if tier == 1 else 1.0,
                "sector": "Synthetic",
            }
        )

    monkeypatch.setattr(opportunities_module, "_BASE_OPPORTUNITIES", synthetic_universe)

    excluded = {"SYN030", "SYN060"}
    threshold = 60.0
    max_results = 5

    start = time.perf_counter()
    output = run_screener_stub(
        exclude_tickers=excluded,
        include_latam=False,
        min_market_cap=1_000,
        max_pe=25.0,
        min_score_threshold=threshold,
        max_results=max_results,
    )
    duration = time.perf_counter() - start

    if isinstance(output, tuple):
        result, notes = output
    else:
        result, notes = output, []

    assert duration < 1.0, "El screener stub debería manejar universos grandes rápidamente"

    assert len(result) == max_results
    assert (result["score_compuesto"] >= threshold).all()
    assert not result["ticker"].isin(excluded).any()
    assert not result["is_latam"].any()

    assert notes, "Se esperaban notas informativas tras el truncado"
    truncation_notes = [note for note in notes if "Se muestran" in note]
    assert truncation_notes, "Debe informarse cuando el resultado se trunca"
    assert str(max_results) in truncation_notes[0]
