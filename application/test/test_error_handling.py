import requests
import pytest

from application.ta_service import fetch_with_indicators
from services import cache


def test_fetch_with_indicators_handles_yfinance_failure(monkeypatch):
    fetch_with_indicators.clear()

    def boom(*args, **kwargs):  # simulate network failure
        raise RuntimeError("fail")

    monkeypatch.setattr("application.ta_service.yf.download", boom)
    with pytest.raises(RuntimeError):
        fetch_with_indicators("AAPL")


def test_fetch_fx_rates_handles_network_error(monkeypatch):
    cache.fetch_fx_rates.clear()

    class FailProv:
        def get_rates(self):
            raise requests.RequestException("boom")

    monkeypatch.setattr(cache, "get_fx_provider", lambda: FailProv())
    data, error = cache.fetch_fx_rates()
    assert data == {}
    assert error is not None
