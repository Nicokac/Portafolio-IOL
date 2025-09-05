import pandas as pd
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
                "revenueGrowth": 0.1,
                "earningsQuarterlyGrowth": 0.2,
            }
            self.sustainability = pd.DataFrame({"Value": [42]}, index=["totalEsg"])
    monkeypatch.setattr(tas, "map_to_us_ticker", fake_map)
    monkeypatch.setattr(tas, "yf", SimpleNamespace(Ticker=lambda t: DummyTicker(t)))
    df = tas.portfolio_fundamentals(["AAA"])
    assert not df.empty
    row = df.iloc[0]
    assert row["symbol"] == "AAA"
    assert row["esg_score"] == 42