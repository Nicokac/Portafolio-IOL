"""Synthetic data tests for TAService and cached helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import application.ta_service as ta_mod
from application.ta_service import TAService


@pytest.fixture(autouse=True)
def _reset_caches():
    """Ensure decorated helpers start from a clean cache for each test."""
    ta_mod.fetch_with_indicators.cache_clear()
    ta_mod.get_portfolio_history.cache_clear()
    ta_mod.get_fundamental_data.cache_clear()
    ta_mod.portfolio_fundamentals.cache_clear()
    yield
    ta_mod.fetch_with_indicators.cache_clear()
    ta_mod.get_portfolio_history.cache_clear()
    ta_mod.get_fundamental_data.cache_clear()
    ta_mod.portfolio_fundamentals.cache_clear()


class _DummyRSI:
    def __init__(self, close: pd.Series, window: int, fillna: bool) -> None:
        self.close = close

    def rsi(self) -> pd.Series:
        return pd.Series(np.linspace(40, 60, len(self.close)), index=self.close.index)


class _DummyMACD:
    def __init__(self, close: pd.Series, window_slow: int, window_fast: int, window_sign: int) -> None:
        self.close = close

    def macd(self) -> pd.Series:
        return pd.Series(np.linspace(-1, 1, len(self.close)), index=self.close.index)

    def macd_signal(self) -> pd.Series:
        return pd.Series(np.zeros(len(self.close)), index=self.close.index)

    def macd_diff(self) -> pd.Series:
        return pd.Series(np.linspace(-0.5, 0.5, len(self.close)), index=self.close.index)


class _DummyATR:
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> None:
        self.high = high
        self.low = low
        self.close = close

    def average_true_range(self) -> pd.Series:
        return pd.Series(np.linspace(1, 2, len(self.close)), index=self.close.index)


class _DummyStochastic:
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int, smooth_window: int) -> None:
        self.close = close

    def stoch(self) -> pd.Series:
        return pd.Series(np.linspace(20, 80, len(self.close)), index=self.close.index)

    def stoch_signal(self) -> pd.Series:
        return pd.Series(np.linspace(15, 70, len(self.close)), index=self.close.index)


class _DummyIchimoku:
    def __init__(self, high: pd.Series, low: pd.Series, window1: int, window2: int, window3: int) -> None:
        self.high = high
        self.low = low

    def ichimoku_conversion_line(self) -> pd.Series:
        return pd.Series(np.linspace(10, 20, len(self.high)), index=self.high.index)

    def ichimoku_base_line(self) -> pd.Series:
        return pd.Series(np.linspace(15, 25, len(self.high)), index=self.high.index)

    def ichimoku_a(self) -> pd.Series:
        return pd.Series(np.linspace(12, 22, len(self.high)), index=self.high.index)

    def ichimoku_b(self) -> pd.Series:
        return pd.Series(np.linspace(8, 18, len(self.high)), index=self.high.index)


def _build_price_frame(periods: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq="D")
    base = np.linspace(100, 120, periods)
    return pd.DataFrame(
        {
            "Open": base + 1,
            "High": base + 2,
            "Low": base - 2,
            "Close": base,
            "Volume": np.linspace(1_000_000, 2_000_000, periods),
        },
        index=idx,
    )


def _setup_indicator_stubs(monkeypatch: pytest.MonkeyPatch, *, frame: pd.DataFrame | None = None):
    frame = frame or _build_price_frame()
    download_calls = {"count": 0}

    def fake_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
        download_calls["count"] += 1
        return frame.copy()

    monkeypatch.setattr(ta_mod, "yf", SimpleNamespace(download=fake_download))
    monkeypatch.setattr(ta_mod, "map_to_us_ticker", lambda sym: sym)
    monkeypatch.setattr(ta_mod, "record_yfinance_usage", lambda *a, **k: None)
    monkeypatch.setattr(ta_mod, "RSIIndicator", _DummyRSI)
    monkeypatch.setattr(ta_mod, "MACD", _DummyMACD)
    monkeypatch.setattr(ta_mod, "AverageTrueRange", _DummyATR)
    monkeypatch.setattr(ta_mod, "StochasticOscillator", _DummyStochastic)
    monkeypatch.setattr(ta_mod, "IchimokuIndicator", _DummyIchimoku)

    return download_calls


def test_indicators_for_returns_enriched_dataframe_and_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _setup_indicator_stubs(monkeypatch)
    svc = TAService()

    df_ind = svc.indicators_for("GGAL", period="3mo", interval="1d")
    assert not df_ind.empty
    expected_columns = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA_FAST",
        "SMA_SLOW",
        "EMA",
        "BB_L",
        "BB_M",
        "BB_U",
        "RSI",
        "MACD",
        "MACD_SIGNAL",
        "MACD_HIST",
        "ATR",
        "STOCH_K",
        "STOCH_D",
        "ICHI_CONV",
        "ICHI_BASE",
        "ICHI_A",
        "ICHI_B",
    }
    assert expected_columns.issubset(df_ind.columns)

    # Cache should prevent a second download for identical parameters
    again = svc.indicators_for("GGAL", period="3mo", interval="1d")
    pd.testing.assert_frame_equal(df_ind, again)
    assert calls["count"] == 1


def test_get_fundamental_data_includes_extended_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    info = {
        "shortName": "TEST",
        "sector": "Tech",
        "website": "https://example.com",
        "marketCap": 1000,
        "trailingPE": 12.0,
        "dividendRate": 0.4,
        "previousClose": 10.0,
        "priceToBook": 1.5,
        "debtToEquity": 1.1,
        "returnOnEquity": 0.18,
        "profitMargins": 0.22,
        "returnOnAssets": 0.09,
        "operatingMargins": 0.3,
        "freeCashflow": 800.0,
        "enterpriseValue": 16000.0,
        "interestCoverage": 6.4,
        "revenueGrowth": 0.12,
        "earningsQuarterlyGrowth": 0.25,
    }

    class DummyTicker:
        def __init__(self, ticker: str) -> None:
            self.info = info.copy()
            self.sustainability = None

    monkeypatch.setattr(ta_mod, "map_to_us_ticker", lambda sym: sym)
    monkeypatch.setattr(ta_mod, "yf", SimpleNamespace(Ticker=lambda t: DummyTicker(t)))

    fundamentals = ta_mod.get_fundamental_data("TEST")
    assert fundamentals["return_on_equity"] == pytest.approx(18.0)
    assert fundamentals["profit_margin"] == pytest.approx(22.0)
    assert fundamentals["return_on_assets"] == pytest.approx(9.0)
    assert fundamentals["operating_margin"] == pytest.approx(30.0)
    assert fundamentals["fcf_yield"] == pytest.approx(5.0)
    assert fundamentals["interest_coverage"] == pytest.approx(6.4)

    df = ta_mod.portfolio_fundamentals(["TEST"])
    assert not df.empty
    row = df.iloc[0]
    assert row["return_on_equity"] == pytest.approx(18.0)
    assert row["profit_margin"] == pytest.approx(22.0)
    assert row["return_on_assets"] == pytest.approx(9.0)
    assert row["operating_margin"] == pytest.approx(30.0)
    assert row["fcf_yield"] == pytest.approx(5.0)
    assert row["interest_coverage"] == pytest.approx(6.4)


def test_portfolio_history_is_cached_and_renames_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.date_range("2024-02-01", periods=5, freq="D")
    hist_df = pd.DataFrame(
        {("GGAL", "Adj Close"): np.linspace(50, 55, len(idx))},
        index=idx,
    )

    calls = {"count": 0}

    def fake_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls["count"] += 1
        return hist_df.copy()

    monkeypatch.setattr(ta_mod, "yf", SimpleNamespace(download=fake_download))
    monkeypatch.setattr(ta_mod, "map_to_us_ticker", lambda sym: sym)
    monkeypatch.setattr(ta_mod, "record_yfinance_usage", lambda *a, **k: None)

    svc = TAService()
    df_first = svc.portfolio_history(simbolos=["GGAL"], period="1mo")
    df_second = svc.portfolio_history(simbolos=["GGAL"], period="1mo")

    assert calls["count"] == 1
    assert list(df_first.columns) == ["GGAL"]
    pd.testing.assert_frame_equal(df_first, df_second)
