import pandas as pd
import pytest
from types import SimpleNamespace
from application import ta_service as tas

def test_portfolio_fundamentals_basic(monkeypatch):
    def fake_map(sym):
        return sym
    class DummyTicker:
        def __init__(self, ticker):
            self.info = {
                "shortName": ticker,
                "sector": "Tech",
                "marketCap": 1000,
                "trailingPE": 15.0,
                "returnOnEquity": 0.12,
                "profitMargins": 0.15,
                "returnOnAssets": 0.09,
                "operatingMargins": 0.2,
                "freeCashflow": 500.0,
                "enterpriseValue": 10000.0,
                "interestCoverage": 4.5,
                "debtToEquity": 1.2,
                "revenueGrowth": 0.1,
                "earningsQuarterlyGrowth": 0.2,
            }
            self.sustainability = pd.DataFrame({"Value": [42]}, index=["totalEsg"])
    monkeypatch.setattr(tas, "map_to_us_ticker", fake_map)
    monkeypatch.setattr(tas, "yf", SimpleNamespace(Ticker=lambda t: DummyTicker(t)))
    tas.portfolio_fundamentals.cache_clear()
    df = tas.portfolio_fundamentals(["AAA"])
    assert not df.empty
    row = df.iloc[0]
    assert row["symbol"] == "AAA"
    assert row["esg_score"] == 42
    assert row["return_on_equity"] == pytest.approx(12.0)
    assert row["profit_margin"] == pytest.approx(15.0)
    assert row["return_on_assets"] == pytest.approx(9.0)
    assert row["operating_margin"] == pytest.approx(20.0)
    assert row["fcf_yield"] == pytest.approx(5.0)
    assert row["interest_coverage"] == pytest.approx(4.5)
