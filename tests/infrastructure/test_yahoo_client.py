from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest
import requests

sys.path.append(str(Path(__file__).resolve().parents[2]))

from infrastructure.market.yahoo_client import YahooFinanceClient, make_symbol_url
from shared.errors import AppError


class SessionFactory:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []

    def __call__(self) -> "FakeSession":
        session = FakeSession()
        self.sessions.append(session)
        return session


class FakeSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeTicker:
    def __init__(self, *, info=None, dividends=None, shares=None, history=None) -> None:
        self._info = info or {}
        self.dividends = dividends
        self._shares = shares
        self._history = history

    def get_info(self):
        return self._info

    def get_shares_full(self, start=None):  # noqa: ARG002
        return self._shares

    def history(self, **kwargs):  # noqa: ARG002
        return self._history


@pytest.fixture(autouse=True)
def clear_client_cache():
    YahooFinanceClient.clear_cache()
    yield
    YahooFinanceClient.clear_cache()


@pytest.fixture
def setup_fake_environment(monkeypatch):
    from infrastructure.market import yahoo_client as module

    YahooFinanceClient.clear_cache()
    session_factory = SessionFactory()
    monkeypatch.setattr(module.requests, "Session", session_factory)

    def factory(*, info=None, dividends=None, shares=None, history=None):
        ticker = FakeTicker(info=info, dividends=dividends, shares=shares, history=history)
        monkeypatch.setattr(module.yf, "Ticker", lambda symbol, session=None: ticker)
        return ticker, session_factory

    return factory


def test_get_fundamentals_normalises_output(setup_fake_environment):
    info = {
        "dividendYield": 0.025,
        "payoutRatio": 0.42,
        "longName": "Test Corp",
        "sector": "Technology",
        "industry": "Software",
        "financialCurrency": "USD",
        "marketCap": 123456,
        "trailingEps": 5.5,
    }
    _, session_factory = setup_fake_environment(info=info)

    client = YahooFinanceClient()
    fundamentals = client.get_fundamentals("tst")

    assert fundamentals["ticker"] == "TST"
    assert fundamentals["dividend_yield"] == pytest.approx(2.5)
    assert fundamentals["payout_ratio"] == pytest.approx(42.0)
    assert fundamentals["market_cap"] == 123456
    assert all(session.closed for session in session_factory.sessions)


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("aapl", "https://finance.yahoo.com/quote/AAPL"),
        ("  nee  ", "https://finance.yahoo.com/quote/NEE"),
        ("", None),
        (None, None),
        (pd.NA, None),
    ],
)
def test_make_symbol_url_normalises_input(symbol, expected):
    assert make_symbol_url(symbol) == expected


def test_invalid_ticker_validation_prevents_request(monkeypatch: pytest.MonkeyPatch):
    from infrastructure.market import yahoo_client as module

    def _unexpected_session():
        raise AssertionError("Session should not be created for invalid ticker")

    monkeypatch.setattr(module.requests, "Session", _unexpected_session)
    monkeypatch.setattr(
        module.yf,
        "Ticker",
        lambda *args, **kwargs: pytest.fail("Ticker should not be requested"),
    )

    client = YahooFinanceClient()

    with pytest.raises(AppError):
        client.get_price_history("bad ticker !")


def test_get_dividends_returns_dataframe(setup_fake_environment):
    dates = pd.to_datetime(["2020-01-01", "2021-01-01"])
    series = pd.Series([0.5, 0.55], index=dates)
    _, session_factory = setup_fake_environment(dividends=series)

    client = YahooFinanceClient()
    df = client.get_dividends("abc")

    assert list(df.columns) == ["date", "amount"]
    assert df["amount"].tolist() == [0.5, 0.55]
    assert df["date"].dt.tz is not None
    assert all(session.closed for session in session_factory.sessions)


def test_get_shares_outstanding_parses_series(setup_fake_environment):
    dates = pd.to_datetime(["2020-01-01", "2022-01-01"])
    series = pd.Series([1_000_000, 950_000], index=dates)
    _, session_factory = setup_fake_environment(shares=series)

    client = YahooFinanceClient()
    df = client.get_shares_outstanding("def")

    assert list(df.columns) == ["date", "shares"]
    assert df["shares"].tolist() == [1_000_000.0, 950_000.0]
    assert df["date"].dt.tz is not None
    assert all(session.closed for session in session_factory.sessions)


def test_get_price_history_validates_columns(setup_fake_environment):
    dates = pd.to_datetime(["2023-01-01", "2023-01-02"])
    history = pd.DataFrame(
        {
            "Close": [10.0, 11.0],
            "Adj Close": [9.5, 10.5],
            "Volume": [1000, 1100],
        },
        index=dates,
    )
    history.index.name = "Date"
    _, session_factory = setup_fake_environment(history=history)

    client = YahooFinanceClient()
    df = client.get_price_history("ghi")

    assert list(df.columns) == ["date", "close", "adj_close", "volume"]
    assert df["close"].tolist() == [10.0, 11.0]
    assert all(session.closed for session in session_factory.sessions)


def test_missing_dividends_raise_app_error(setup_fake_environment):
    _, _ = setup_fake_environment(dividends=pd.Series(dtype=float))
    client = YahooFinanceClient()

    with pytest.raises(AppError):
        client.get_dividends("xyz")


def test_get_fundamentals_uses_persistent_cache(setup_fake_environment):
    info = {
        "dividendYield": 0.02,
        "payoutRatio": 0.3,
        "marketCap": 10_000,
    }
    _, session_factory = setup_fake_environment(info=info)

    client = YahooFinanceClient()
    client.get_fundamentals("cache")
    initial_sessions = len(session_factory.sessions)

    client.get_fundamentals("cache")
    assert len(session_factory.sessions) == initial_sessions

    another_client = YahooFinanceClient()
    another_client.get_fundamentals("cache")
    assert len(session_factory.sessions) == initial_sessions


def test_http_404_downgraded_and_cached(monkeypatch: pytest.MonkeyPatch, caplog):
    from infrastructure.market import yahoo_client as module

    session_factory = SessionFactory()
    monkeypatch.setattr(module.requests, "Session", session_factory)

    response = requests.Response()
    response.status_code = 404
    error = requests.HTTPError("not found", response=response)

    class RaisingTicker(FakeTicker):
        def get_info(self):
            raise error

    monkeypatch.setattr(module.yf, "Ticker", lambda symbol, session=None: RaisingTicker())

    client = YahooFinanceClient()

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(AppError):
            client.get_fundamentals("missing")

    debug_messages = [
        record.message for record in caplog.records if record.levelno == logging.DEBUG
    ]
    assert any("Yahoo Finance devolvió 404" in message for message in debug_messages)
    assert len(session_factory.sessions) == 1

    caplog.clear()

    with pytest.raises(AppError):
        client.get_fundamentals("missing")

    assert len(session_factory.sessions) == 1
