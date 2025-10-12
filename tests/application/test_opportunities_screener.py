import pandas as pd
import pytest

from application.screener import opportunities as ops


class DummyFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class DummyExecutor:
    def __init__(self, *, max_workers: int):
        self.max_workers = max_workers
        self.submissions: list[tuple] = []

    def submit(self, func, *args, **kwargs):
        self.submissions.append((func, args, kwargs))
        value = func(*args, **kwargs)
        return DummyFuture(value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def fake_as_completed(futures):
    return list(futures)


def test_precheck_symbols_discards_low_quality():
    listings_meta = {
        "AAA": {"market_cap": 100_000_000, "pe_ratio": 120, "revenue_growth": -5},
        "BBB": {"market_cap": 750_000_000, "pe_ratio": 25, "revenue_growth": 2},
    }
    filtered, dropped, ratio = ops._precheck_symbols(
        ["AAA", "BBB"],
        listings_meta=listings_meta,
        min_market_cap=None,
        max_pe=None,
        min_revenue_growth=None,
    )

    assert filtered == ["BBB"]
    assert dropped == {"AAA": ["market_cap<500,000,000", "pe_ratio>50", "revenue_growth<0.00"]}
    assert pytest.approx(ratio, rel=1e-3) == 0.5


def test_run_screener_yahoo_uses_thread_pool(monkeypatch):
    executors: list[DummyExecutor] = []

    def _executor_factory(*, max_workers: int):
        executor = DummyExecutor(max_workers=max_workers)
        executors.append(executor)
        return executor

    monkeypatch.setattr(ops, "ThreadPoolExecutor", _executor_factory)
    monkeypatch.setattr(ops, "as_completed", fake_as_completed)

    class FakeClient:
        def __init__(self) -> None:
            self.calls: dict[str, int] = {}

        def _touch(self, name: str) -> None:
            self.calls[name] = self.calls.get(name, 0) + 1

        def get_fundamentals(self, ticker: str):
            self._touch(f"fundamentals:{ticker}")
            return {
                "ticker": ticker,
                "payout_ratio": 0.4,
                "dividend_yield": 1.5,
                "market_cap": 800_000_000,
            }

        def get_dividends(self, ticker: str):
            self._touch(f"dividends:{ticker}")
            dates = pd.to_datetime(["2020-01-01", "2021-01-01"])
            return pd.DataFrame({"date": dates, "amount": [0.5, 0.6]})

        def get_shares_outstanding(self, ticker: str):
            self._touch(f"shares:{ticker}")
            dates = pd.to_datetime(["2020-01-01", "2021-01-01"])
            return pd.DataFrame({"date": dates, "shares": [1_000_000, 950_000]})

        def get_price_history(self, ticker: str):
            self._touch(f"prices:{ticker}")
            dates = pd.to_datetime(["2020-01-01", "2020-01-02"])
            return pd.DataFrame(
                {
                    "date": dates,
                    "close": [10.0, 11.0],
                    "adj_close": [9.5, 10.5],
                    "volume": [1000, 1100],
                }
            )

    client = FakeClient()
    result = ops.run_screener_yahoo(manual_tickers=["AAA", "BBB"], client=client)
    if isinstance(result, tuple):
        df, _notes = result
    else:
        df = result

    assert executors, "expected ThreadPoolExecutor to be used"
    assert executors[0].max_workers == ops._MAX_TICKER_WORKERS

    summary = df.attrs.get("summary", {})
    assert summary.get("precheck_initial_count") == 2
    assert summary.get("precheck_discard_ratio") == 0.0
    elapsed_map = summary.get("elapsed_seconds_per_ticker")
    assert isinstance(elapsed_map, dict)
    assert set(elapsed_map.keys()) == {"AAA", "BBB"}
