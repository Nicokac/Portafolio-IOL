"""Tests for the opportunities controller contract with the UI."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers import opportunities as sut
from shared.errors import AppError


_EXPECTED_COLUMNS = [
    "ticker",
    "payout_ratio",
    "dividend_streak",
    "cagr",
    "dividend_yield",
    "price",
    "score_compuesto",
]
_EXPECTED_WITH_TECHNICALS = _EXPECTED_COLUMNS + ["rsi", "sma_50", "sma_200"]


@pytest.fixture(autouse=True)
def restore_yahoo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests start with a callable Yahoo screener and isolated stub."""

    # Each test will set the behaviour explicitly; default to raising so
    # accidental usage is caught quickly.
    monkeypatch.setattr(
        sut,
        "run_screener_yahoo",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("not patched")),
    )
    monkeypatch.setattr(
        sut,
        "run_screener_stub",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("not patched")),
    )


def _make_sample_row(include_technicals: bool = False) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "ticker": "AAPL",
        "payout_ratio": 25.0,
        "dividend_streak": 10,
        "cagr": 8.0,
        "dividend_yield": 0.8,
        "price": 170.0,
        "score_compuesto": 75.0,
    }
    if include_technicals:
        row.update({"rsi": 55.0, "sma_50": 160.0, "sma_200": 150.0})
    return row


def test_propagates_filters_and_uses_yahoo(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs: Dict[str, Any] = {}

    def fake_yahoo(**kwargs: Any) -> pd.DataFrame:
        captured_kwargs.clear()
        captured_kwargs.update(kwargs)
        base = _make_sample_row(include_technicals=True)
        extra = {**base, "ticker": "MSFT"}
        return pd.DataFrame([base, extra])

    monkeypatch.setattr(sut, "run_screener_yahoo", fake_yahoo)
    monkeypatch.setattr(
        sut,
        "run_screener_stub",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("fallback used")),
    )

    df, notes = sut.run_opportunities_controller(
        manual_tickers=[" aapl", "MSFT", "msft"],
        max_payout=70.0,
        min_div_streak=5,
        min_cagr=3.5,
        include_technicals=True,
        min_market_cap=1_500_000_000,
        max_pe=25,
        min_revenue_growth=7.5,
        include_latam=True,
    )

    assert captured_kwargs == {
        "manual_tickers": ["AAPL", "MSFT"],
        "include_technicals": True,
        "max_payout": pytest.approx(70.0),
        "min_div_streak": 5,
        "min_cagr": pytest.approx(3.5),
        "min_market_cap": pytest.approx(1_500_000_000.0),
        "max_pe": pytest.approx(25.0),
        "min_revenue_growth": pytest.approx(7.5),
        "include_latam": True,
    }
    assert list(df.columns) == _EXPECTED_WITH_TECHNICALS
    assert notes == []


def test_fallback_to_stub_preserves_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sut,
        "run_screener_yahoo",
        lambda **kwargs: (_ for _ in ()).throw(AppError("boom")),
    )

    stub_result = pd.DataFrame([{"ticker": "AAPL"}])
    stub_calls: Dict[str, Any] = {}

    def fake_stub(**kwargs: Any) -> pd.DataFrame:
        stub_calls.clear()
        stub_calls.update(kwargs)
        return stub_result

    monkeypatch.setattr(sut, "run_screener_stub", fake_stub)

    df, notes = sut.run_opportunities_controller(
        manual_tickers=["aapl", None],
        max_payout=60,
        min_div_streak=8,
        min_cagr=4.2,
        include_technicals=False,
    )

    assert stub_calls["manual_tickers"] == ["AAPL"]
    assert stub_calls["max_payout"] == 60
    assert stub_calls["min_div_streak"] == 8
    assert stub_calls["min_cagr"] == 4.2
    assert stub_calls["include_technicals"] is False
    assert list(df.columns) == _EXPECTED_COLUMNS
    assert notes[0] == "⚠️ Datos simulados (Yahoo no disponible)"
    assert "AAPL" in notes[1]


def test_normalises_incomplete_yahoo_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "data": [
            {"ticker": "AAA", "price": 10.0},
            {"ticker": "CCC", "dividend_yield": 1.5},
        ],
        "notes": ["Partial data"],
    }

    monkeypatch.setattr(sut, "run_screener_yahoo", lambda **kwargs: payload)
    monkeypatch.setattr(
        sut,
        "run_screener_stub",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("fallback used")),
    )

    df, notes = sut.run_opportunities_controller(
        manual_tickers=["aaa", "bbb"],
        include_technicals=True,
    )

    assert list(df.columns) == _EXPECTED_WITH_TECHNICALS
    aaa_row = df[df["ticker"] == "AAA"].iloc[0]
    assert aaa_row["price"] == pytest.approx(10.0)
    assert pd.isna(aaa_row["payout_ratio"])
    assert pd.isna(aaa_row["dividend_streak"])
    assert pd.isna(aaa_row["rsi"])
    assert "Partial data" in notes
    assert any("BBB" in note for note in notes)
