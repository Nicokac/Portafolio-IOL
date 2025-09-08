import pandas as pd


from application.ta_service import fetch_with_indicators


def test_fetch_with_indicators_handles_yfinance_failure(monkeypatch):
    fetch_with_indicators.clear()

    def boom(*args, **kwargs):  # simulate network failure
        raise RuntimeError("fail")

    monkeypatch.setattr("application.ta_service.yf.download", boom)
    df = fetch_with_indicators("AAPL")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


# def test_fetch_fx_rates_handles_failure(monkeypatch):
#     import app
from services import cache

    # app.fetch_fx_rates.clear()
def test_fetch_fx_rates_handles_failure(monkeypatch):
    cache.fetch_fx_rates.clear()

    class FailProv:
        def get_rates(self):
            raise RuntimeError("boom")

    # monkeypatch.setattr(app, "get_fx_provider", lambda: FailProv())
    # assert app.fetch_fx_rates() == {}
    monkeypatch.setattr(cache, "get_fx_provider", lambda: FailProv())
    assert cache.fetch_fx_rates() == {}