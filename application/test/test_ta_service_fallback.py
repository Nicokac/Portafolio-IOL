from __future__ import annotations

import pandas as pd
import pytest

from application.ta_service import fetch_with_indicators


class _DummyRSI:
    def __init__(self, close: pd.Series, window: int, fillna: bool) -> None:
        self._index = close.index

    def rsi(self) -> pd.Series:
        return pd.Series([50.0] * len(self._index), index=self._index)


class _DummyMACD:
    def __init__(self, close: pd.Series, window_slow: int, window_fast: int, window_sign: int) -> None:
        self._index = close.index

    def macd(self) -> pd.Series:
        return pd.Series([0.0] * len(self._index), index=self._index)

    def macd_signal(self) -> pd.Series:
        return pd.Series([0.0] * len(self._index), index=self._index)

    def macd_diff(self) -> pd.Series:
        return pd.Series([0.0] * len(self._index), index=self._index)


class _DummyATR:
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> None:
        self._index = close.index

    def average_true_range(self) -> pd.Series:
        return pd.Series([1.0] * len(self._index), index=self._index)


class _DummyStoch:
    def __init__(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int,
        smooth_window: int,
    ) -> None:
        self._index = close.index

    def stoch(self) -> pd.Series:
        return pd.Series([20.0] * len(self._index), index=self._index)

    def stoch_signal(self) -> pd.Series:
        return pd.Series([25.0] * len(self._index), index=self._index)


class _DummyIchimoku:
    def __init__(self, high: pd.Series, low: pd.Series, window1: int, window2: int, window3: int) -> None:
        self._index = high.index

    def ichimoku_conversion_line(self) -> pd.Series:
        return pd.Series([10.0] * len(self._index), index=self._index)

    def ichimoku_base_line(self) -> pd.Series:
        return pd.Series([15.0] * len(self._index), index=self._index)

    def ichimoku_a(self) -> pd.Series:
        return pd.Series([12.0] * len(self._index), index=self._index)

    def ichimoku_b(self) -> pd.Series:
        return pd.Series([8.0] * len(self._index), index=self._index)


class _AdapterStub:
    def __init__(self, payload: pd.DataFrame | Exception) -> None:
        self.payload = payload
        self.calls = 0

    def fetch(self, symbol: str, **kwargs):
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload.copy()


@pytest.fixture(autouse=True)
def _reset_fetch_cache():
    fetch_with_indicators.clear()
    yield
    fetch_with_indicators.clear()


def _patch_indicators(monkeypatch: pytest.MonkeyPatch, adapter: _AdapterStub) -> None:
    monkeypatch.setattr("application.ta_service.get_ohlc_adapter", lambda: adapter)
    monkeypatch.setattr("application.ta_service.map_to_us_ticker", lambda sym: sym)
    monkeypatch.setattr("application.ta_service.RSIIndicator", _DummyRSI)
    monkeypatch.setattr("application.ta_service.MACD", _DummyMACD)
    monkeypatch.setattr("application.ta_service.AverageTrueRange", _DummyATR)
    monkeypatch.setattr("application.ta_service.StochasticOscillator", _DummyStoch)
    monkeypatch.setattr("application.ta_service.IchimokuIndicator", _DummyIchimoku)


def test_fetch_with_indicators_returns_data_when_adapter_succeeds(
    monkeypatch: pytest.MonkeyPatch,
):
    periods = 60
    idx = pd.date_range("2024-01-01", periods=periods, freq="D")
    frame = pd.DataFrame(
        {
            "Open": pd.Series(range(100, 100 + periods), index=idx, dtype="float64"),
            "High": pd.Series(range(101, 101 + periods), index=idx, dtype="float64"),
            "Low": pd.Series(range(99, 99 + periods), index=idx, dtype="float64"),
            "Close": pd.Series(range(100, 100 + periods), index=idx, dtype="float64"),
            "Volume": pd.Series([1_000_000 + i for i in range(periods)], index=idx, dtype="float64"),
        }
    )
    adapter = _AdapterStub(frame)
    _patch_indicators(monkeypatch, adapter)

    result = fetch_with_indicators("AAPL")
    assert not result.empty
    assert adapter.calls == 1


def test_fetch_with_indicators_returns_empty_when_adapter_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = _AdapterStub(pd.DataFrame())
    _patch_indicators(monkeypatch, adapter)

    result = fetch_with_indicators("AAPL")
    assert result.empty
    assert adapter.calls == 1


def test_fetch_with_indicators_raises_runtime_error_on_adapter_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = _AdapterStub(RuntimeError("fail"))
    _patch_indicators(monkeypatch, adapter)

    with pytest.raises(RuntimeError):
        fetch_with_indicators("AAPL")
    assert adapter.calls == 1
