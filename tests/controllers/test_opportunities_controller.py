"""Tests for the opportunities controller contract with the UI."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    "sector",
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
        "sector": "Technology",
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

    def fake_yahoo(**kwargs: Any) -> Tuple[pd.DataFrame, List[str]]:
        captured_kwargs.clear()
        captured_kwargs.update(kwargs)
        base = _make_sample_row(include_technicals=True)
        extra = {**base, "ticker": "MSFT"}
        df = pd.DataFrame([base, extra])
        exclude = set(kwargs.get("exclude_tickers") or [])
        if exclude:
            df = df[~df["ticker"].isin(exclude)]
        return df, ["Yahoo note"]

    monkeypatch.setattr(sut, "run_screener_yahoo", fake_yahoo)
    monkeypatch.setattr(
        sut,
        "run_screener_stub",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("fallback used")),
    )

    df, notes, source = sut.run_opportunities_controller(
        manual_tickers=[" aapl", "MSFT", "msft"],
        exclude_tickers=[" msft"],
        max_payout=70.0,
        min_div_streak=5,
        min_cagr=3.5,
        sectors=["technology", "HEALTHCARE"],
        include_technicals=True,
        min_market_cap=1_500_000_000,
        max_pe=25,
        min_revenue_growth=7.5,
        include_latam=True,
        min_eps_growth=4.5,
        min_buyback=0.5,
        min_score_threshold=42.5,
        max_results=15,
    )

    assert captured_kwargs == {
        "manual_tickers": ["AAPL", "MSFT"],
        "include_technicals": True,
        "exclude_tickers": ["MSFT"],
        "max_payout": pytest.approx(70.0),
        "min_div_streak": 5,
        "min_cagr": pytest.approx(3.5),
        "min_market_cap": pytest.approx(1_500_000_000.0),
        "max_pe": pytest.approx(25.0),
        "min_revenue_growth": pytest.approx(7.5),
        "include_latam": True,
        "min_eps_growth": pytest.approx(4.5),
        "min_buyback": pytest.approx(0.5),
        "min_score_threshold": pytest.approx(42.5),
        "max_results": 15,
        "sectors": ["Technology", "Healthcare"],
    }
    assert list(df.columns) == _EXPECTED_WITH_TECHNICALS
    assert set(df["ticker"]) == {"AAPL"}
    assert notes == ["Yahoo note"]
    assert source == "yahoo"


def test_controller_propagates_yahoo_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = pd.DataFrame([_make_sample_row()])

    def fake_yahoo(**kwargs: Any) -> Tuple[pd.DataFrame, List[str]]:  # noqa: ARG001
        return payload, ["Nota desde Yahoo"]

    monkeypatch.setattr(sut, "run_screener_yahoo", fake_yahoo)
    monkeypatch.setattr(
        sut,
        "run_screener_stub",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("stub used")),
    )

    df, notes, source = sut.run_opportunities_controller(
        manual_tickers=["aapl"],
        include_technicals=False,
    )

    assert df.equals(payload[_EXPECTED_COLUMNS])
    assert notes == ["Nota desde Yahoo"]
    assert source == "yahoo"


def test_fallback_to_stub_preserves_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sut,
        "run_screener_yahoo",
        lambda **kwargs: (_ for _ in ()).throw(AppError("boom")),
    )

    stub_result = pd.DataFrame([
        {"ticker": "AAPL"},
        {"ticker": "MSFT"},
    ])
    stub_calls: Dict[str, Any] = {}

    def fake_stub(**kwargs: Any) -> Tuple[pd.DataFrame, List[str]]:
        stub_calls.clear()
        stub_calls.update(kwargs)
        exclude = set(kwargs.get("exclude_tickers") or [])
        if not exclude:
            return stub_result, ["Stub note"]
        filtered = stub_result[~stub_result["ticker"].isin(exclude)]
        return filtered, ["Stub note"]
    monkeypatch.setattr(sut, "run_screener_stub", fake_stub)

    df, notes, source = sut.run_opportunities_controller(
        manual_tickers=["aapl", None],
        exclude_tickers=["MSFT"],
        max_payout=60,
        min_div_streak=8,
        min_cagr=4.2,
        include_technicals=False,
        min_eps_growth=2.0,
        min_buyback=0.0,
        min_score_threshold="30.0",
        max_results="2",
    )

    assert stub_calls["manual_tickers"] == ["AAPL"]
    assert stub_calls["exclude_tickers"] == ["MSFT"]
    assert stub_calls["max_payout"] == 60
    assert stub_calls["min_div_streak"] == 8
    assert stub_calls["min_cagr"] == 4.2
    assert stub_calls["include_technicals"] is False
    assert stub_calls["min_eps_growth"] == 2.0
    assert stub_calls["min_buyback"] == 0.0
    assert stub_calls["min_score_threshold"] == 30.0
    assert stub_calls["max_results"] == 2
    assert "sector" in df.columns
    assert stub_calls["sectors"] is None
    assert list(df.columns) == _EXPECTED_COLUMNS
    assert "MSFT" not in set(df["ticker"])
    assert notes[0] == "⚠️ Datos simulados (Yahoo no disponible)"
    assert notes[1] == "Stub note"
    assert "AAPL" in notes[2]
    assert source == "stub"


def test_excluded_tickers_are_removed_from_stub_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sut,
        "run_screener_yahoo",
        lambda **kwargs: (_ for _ in ()).throw(AppError("boom")),
    )

    def fake_stub(**kwargs: Any) -> Tuple[pd.DataFrame, List[str]]:
        data = pd.DataFrame([
            {"ticker": "AAPL", "sector": "Technology"},
            {"ticker": "MSFT", "sector": "Technology"},
        ])
        exclude = set(kwargs.get("exclude_tickers") or [])
        if exclude:
            data = data[~data["ticker"].isin(exclude)]
        return data, []

    monkeypatch.setattr(sut, "run_screener_stub", fake_stub)

    df, notes, source = sut.run_opportunities_controller(
        manual_tickers=["AAPL", "MSFT"],
        exclude_tickers=["MSFT"],
        min_buyback=0.0,
    )

    assert "MSFT" not in set(df["ticker"])
    assert any(ticker == "AAPL" for ticker in df["ticker"])
    assert not any("MSFT" in note for note in notes)
    assert source == "stub"


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

    df, notes, source = sut.run_opportunities_controller(
        manual_tickers=["aaa", "bbb"],
        include_technicals=True,
    )

    assert list(df.columns) == _EXPECTED_WITH_TECHNICALS
    aaa_row = df[df["ticker"] == "AAA"].iloc[0]
    assert aaa_row["price"] == pytest.approx(10.0)
    assert pd.isna(aaa_row["payout_ratio"])
    assert pd.isna(aaa_row["dividend_streak"])
    assert pd.isna(aaa_row["sector"])
    assert pd.isna(aaa_row["rsi"])
    assert "Partial data" in notes
    assert any("BBB" in note for note in notes)
    assert source == "yahoo"


def test_generate_report_includes_source(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame([_make_sample_row()])

    def fake_controller(**kwargs: Any) -> Tuple[pd.DataFrame, List[str], str]:
        assert kwargs["manual_tickers"] == ["abc"]
        assert kwargs["include_technicals"] is True
        assert kwargs.get("sectors") is None
        assert kwargs.get("exclude_tickers") is None
        assert kwargs["min_score_threshold"] == pytest.approx(35.5)
        assert kwargs["max_results"] == 25
        return df, ["note"], "stub"

    monkeypatch.setattr(sut, "run_opportunities_controller", fake_controller)

    result = sut.generate_opportunities_report(
        {
            "manual_tickers": ["abc"],
            "include_technicals": True,
            "min_score_threshold": "35.5",
            "max_results": "25",
        }
    )

    assert result == {"table": df, "notes": ["note"], "source": "stub"}


def test_generate_report_parses_string_bool(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame([_make_sample_row()])

    captured: Dict[str, Any] = {}

    def fake_controller(**kwargs: Any) -> Tuple[pd.DataFrame, List[str], str]:
        captured.update(kwargs)
        return df, [], "yahoo"

    monkeypatch.setattr(sut, "run_opportunities_controller", fake_controller)

    sut.generate_opportunities_report({"include_technicals": "false"})

    assert captured["include_technicals"] is False
    assert "exclude_tickers" not in captured or captured["exclude_tickers"] is None
