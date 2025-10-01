from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from application.screener.opportunities import run_screener_stub
import application.screener.opportunities as opportunities_module
from shared import settings as shared_settings


_CRITICAL_SECTORS = (
    "Technology",
    "Energy",
    "Industrials",
    "Consumer",
    "Healthcare",
    "Financials",
    "Utilities",
    "Materials",
)


def _tickers(df: pd.DataFrame) -> set[str]:
    return set(df["ticker"].dropna().astype(str))


def _run_stub_with_notes(**kwargs: object) -> tuple[pd.DataFrame, list[str]]:
    result = run_screener_stub(**kwargs)
    if isinstance(result, tuple):
        df, notes = result
    else:
        df, notes = result, []
    return df, notes


def _find_stub_note(notes: list[str]) -> str:
    for note in notes:
        if note.startswith("ℹ️ Stub procesó ") or note.startswith("⚠️ Stub procesó "):
            if " segundos" in note:
                return note
    raise AssertionError("Se esperaba una nota de telemetría del stub")


def _assert_has_stub_note(notes: list[str]) -> None:
    _find_stub_note(notes)


def test_filters_apply_to_base_dataset() -> None:
    df, notes = _run_stub_with_notes(
        min_market_cap=100_000,
        max_pe=35.0,
        min_revenue_growth=5.0,
        include_latam=False,
    )
    _assert_has_stub_note(notes)
    telemetry_note = _find_stub_note(notes)
    base = pd.DataFrame(opportunities_module._BASE_OPPORTUNITIES)
    base_filtered = base[
        (base["market_cap"] >= 100_000)
        & (base["pe_ratio"] <= 35.0)
        & (base["revenue_growth"] >= 5.0)
        & (~base["is_latam"])
    ]

    assert _tickers(df) == set(base_filtered["ticker"].unique())
    assert {"ticker", "sector", "score_compuesto"}.issubset(df.columns)
    assert (df["market_cap"] >= 100_000).all()
    assert (df["pe_ratio"] <= 35.0).all()
    assert (df["revenue_growth"] >= 5.0).all()
    assert not df.get("is_latam", pd.Series(dtype=bool)).any()
    assert {"rsi", "sma_50", "sma_200"}.isdisjoint(df.columns)
    for column in [
        "payout_ratio",
        "dividend_streak",
        "cagr",
        "dividend_yield",
        "buyback",
    ]:
        assert base_filtered[column].notna().all(), f"Expected column '{column}' to have values in the base dataset"
    for column in [
        "trailing_eps",
        "forward_eps",
    ]:
        assert base_filtered[column].notna().all(), f"Expected column '{column}' to have values in the base dataset"

    assert f"(resultado: {len(df)})" in telemetry_note
    assert "descartes:" in telemetry_note
    for filter_key in ("min_market_cap", "max_pe", "min_revenue_growth", "include_latam"):
        label = filter_key
        if filter_key == "include_latam":
            label = "include_latam=False"
        pattern = rf"{label}: \d+/\d+ \(\d+%\)"
        assert re.search(pattern, telemetry_note), telemetry_note


def test_include_latam_flag_keeps_companies_when_enabled() -> None:
    df, notes = _run_stub_with_notes(include_latam=True, include_technicals=True)
    _assert_has_stub_note(notes)
    latam_tickers = {
        row["ticker"]
        for row in opportunities_module._BASE_OPPORTUNITIES
        if row.get("is_latam")
    }
    assert latam_tickers.issubset(_tickers(df))
    for ticker in latam_tickers:
        assert bool(df.loc[df["ticker"] == ticker, "is_latam"].iloc[0])
    assert {"rsi", "sma_50", "sma_200"}.issubset(df.columns)


def test_manual_tickers_return_placeholder_after_filters() -> None:
    df, notes = _run_stub_with_notes(
        manual_tickers=["MELI"],
        include_latam=False,
        max_pe=20.0,
    )
    _assert_has_stub_note(notes)
    assert "MELI" in _tickers(df)
    # MELI is excluded by filters, so its numeric fields should be NaN after placeholder padding.
    row = df.loc[df["ticker"] == "MELI"].iloc[0]
    assert pd.isna(row["pe_ratio"])
    assert pd.isna(row["market_cap"])


def test_run_screener_stub_applies_score_threshold_inclusively() -> None:
    baseline, baseline_notes = _run_stub_with_notes(include_technicals=False)
    _assert_has_stub_note(baseline_notes)
    ordered = baseline.sort_values("score_compuesto", ascending=True).reset_index(drop=True)

    below_row = ordered.iloc[0]
    threshold_row = ordered.iloc[1]
    threshold_value = float(threshold_row["score_compuesto"])

    assert threshold_value > float(below_row["score_compuesto"])

    filtered, filtered_notes = _run_stub_with_notes(
        include_technicals=False,
        min_score_threshold=threshold_value,
    )
    _assert_has_stub_note(filtered_notes)

    assert threshold_row["ticker"] in _tickers(filtered)
    assert below_row["ticker"] not in _tickers(filtered)
    assert (filtered["score_compuesto"] >= threshold_value).all()


def test_stub_dataset_is_diverse_and_complete() -> None:
    base = pd.DataFrame(opportunities_module._BASE_OPPORTUNITIES)
    assert len(base) >= len(_CRITICAL_SECTORS) * 3
    assert len(base) <= 45
    assert base["sector"].nunique() >= 10
    sector_counts = base["sector"].value_counts()
    for sector in _CRITICAL_SECTORS:
        count = int(sector_counts.get(sector, 0))
        assert count >= 3, f"El sector crítico '{sector}' debería tener al menos 3 emisores"
        assert count <= 5, f"El sector crítico '{sector}' debería mantenerse acotado para el QA"
    latam_rows = base[base["is_latam"]]
    assert len(latam_rows) >= 3
    for column in [
        "market_cap",
        "pe_ratio",
        "revenue_growth",
        "trailing_eps",
        "forward_eps",
        "buyback",
        "dividend_streak",
        "cagr",
    ]:
        assert base[column].notna().all(), f"Column '{column}' should be populated in the stub"
    assert (base["market_cap"] > 0).all()


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
    result, notes = _run_stub_with_notes(
        exclude_tickers=excluded,
        include_latam=False,
        min_market_cap=1_000,
        max_pe=25.0,
        min_score_threshold=threshold,
        max_results=max_results,
    )
    duration = time.perf_counter() - start

    assert duration < 1.0, "El screener stub debería manejar universos grandes rápidamente"

    assert len(result) == max_results
    assert (result["score_compuesto"] >= threshold).all()
    assert not result["ticker"].isin(excluded).any()
    assert not result["is_latam"].any()

    assert notes, "Se esperaban notas informativas tras el truncado"
    truncation_notes = [note for note in notes if "Se muestran" in note]
    assert truncation_notes, "Debe informarse cuando el resultado se trunca"
    assert str(max_results) in truncation_notes[0]
    _assert_has_stub_note(notes)


def test_run_screener_stub_emits_telemetry_note() -> None:
    _df, notes = _run_stub_with_notes()
    _assert_has_stub_note(notes)


def test_run_screener_stub_emits_info_severity_under_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = iter([100.0, 100.05])

    def _fake_perf_counter() -> float:
        try:
            return next(calls)
        except StopIteration:
            return 100.05

    monkeypatch.setattr(opportunities_module.time, "perf_counter", _fake_perf_counter)
    monkeypatch.setattr(shared_settings, "STUB_MAX_RUNTIME_WARN", 0.2, raising=False)

    df, notes = _run_stub_with_notes()
    note = _find_stub_note(notes)

    assert note.startswith("ℹ️ ")
    assert df.attrs["_notes"][-1] == note


def test_run_screener_stub_emits_warning_severity_over_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = iter([200.0, 200.5])

    def _fake_perf_counter() -> float:
        try:
            return next(calls)
        except StopIteration:
            return 200.5

    monkeypatch.setattr(opportunities_module.time, "perf_counter", _fake_perf_counter)
    monkeypatch.setattr(shared_settings, "STUB_MAX_RUNTIME_WARN", 0.25, raising=False)

    _df, notes = _run_stub_with_notes()
    note = _find_stub_note(notes)

    assert note.startswith("⚠️ ")
