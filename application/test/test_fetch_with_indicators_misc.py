import pandas as pd
import pytest

from application.ta_service import fetch_with_indicators


def _empty_df():
    return pd.DataFrame()


def _sample_df():
    index = pd.date_range("2023-01-01", periods=60)
    return pd.DataFrame(
        {
            "Open": range(60),
            "High": range(60),
            "Low": range(60),
            "Close": range(60),
            "Volume": [100] * 60,
        },
        index=index,
    )


def test_raises_when_yfinance_missing(monkeypatch):
    fetch_with_indicators.clear()
    monkeypatch.setattr("application.ta_service.yf", None)
    monkeypatch.setattr("application.ta_service.map_to_us_ticker", lambda s: "AAPL")
    with pytest.raises(RuntimeError):
        fetch_with_indicators("AAPL")


def test_returns_empty_on_empty_download(monkeypatch):
    fetch_with_indicators.clear()
    monkeypatch.setattr("application.ta_service.map_to_us_ticker", lambda s: "AAPL")
    monkeypatch.setattr("application.ta_service.yf.download", lambda *a, **k: _empty_df())
    df = fetch_with_indicators("AAPL")
    assert df.empty


def test_lru_cache_downloads_once(monkeypatch):
    fetch_with_indicators.clear()
    monkeypatch.setattr("application.ta_service.map_to_us_ticker", lambda s: "AAPL")
    calls = {"n": 0}

    def fake_download(*args, **kwargs):
        calls["n"] += 1
        return _sample_df()

    monkeypatch.setattr("application.ta_service.yf.download", fake_download)
    df1 = fetch_with_indicators("AAPL")
    df2 = fetch_with_indicators("AAPL")
    assert calls["n"] == 1
    assert not df1.empty and not df2.empty
