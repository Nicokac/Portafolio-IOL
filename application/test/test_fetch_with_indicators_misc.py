import pandas as pd

import pytest

from application.ta_service import TAService, fetch_with_indicators


def _hist_df():
    index = pd.date_range("2023-01-01", periods=60)
    return pd.DataFrame({
        "Open": range(1, 61),
        "High": range(2, 62),
        "Low": range(0, 60),
        "Close": range(1, 61),
        "Volume": [100] * 60,
    }, index=index)


def test_indicators_for_valid_symbol(monkeypatch):
    fetch_with_indicators.clear()
    monkeypatch.setattr("application.ta_service.yf.download", lambda *a, **k: _hist_df())
    monkeypatch.setattr("application.ta_service.map_to_us_ticker", lambda s: "AAPL")
    svc = TAService()
    df = svc.indicators_for("AAPL")
    assert not df.empty
    assert {"SMA_FAST", "SMA_SLOW", "RSI"}.issubset(df.columns)


def test_indicators_for_invalid_symbol(monkeypatch):
    fetch_with_indicators.clear()
    monkeypatch.setattr("application.ta_service.map_to_us_ticker", lambda s: None)
    def bad_map(sym):
        raise ValueError("bad")
    monkeypatch.setattr("application.ta_service.map_to_us_ticker", bad_map)
    svc = TAService()
    with pytest.raises(ValueError):
        svc.indicators_for("INVALID")