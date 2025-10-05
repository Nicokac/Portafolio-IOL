from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from application.portfolio_service import PortfolioService
from controllers.portfolio import filters
from domain.models import Controls
from infrastructure.iol import client as iol_client_module
from services import cache as cache_module


def _http_error(status: int) -> requests.HTTPError:
    response = SimpleNamespace(status_code=status)
    return requests.HTTPError(response=response)


class FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict = {}
        self.errors: list[str] = []

    def error(self, message: str) -> None:  # pragma: no cover - defensive
        self.errors.append(str(message))

    def warning(self, message: str) -> None:  # pragma: no cover - defensive
        self.errors.append(str(message))

    def stop(self) -> None:  # pragma: no cover - defensive
        raise AssertionError("stop() should not be called in tests")


@pytest.fixture
def fake_streamlit(monkeypatch: pytest.MonkeyPatch) -> FakeStreamlit:
    fake_st = FakeStreamlit()
    for module in (filters, cache_module, iol_client_module):
        monkeypatch.setattr(module, "st", fake_st)
    monkeypatch.setattr("shared.cache.st", fake_st, raising=False)
    return fake_st


def test_portfolio_flow_recovers_via_ohlc_after_legacy_429(
    monkeypatch: pytest.MonkeyPatch, fake_streamlit: FakeStreamlit
) -> None:
    cache_module.fetch_quotes_bulk.clear()
    cache_module._QUOTE_CACHE.clear()
    monkeypatch.setattr(cache_module, "_persist_quote", lambda *_, **__: None)
    monkeypatch.setattr(cache_module, "_load_persisted_entry", lambda key: None)
    monkeypatch.setattr(cache_module, "record_quote_provider_usage", lambda *_, **__: None)
    monkeypatch.setattr(cache_module, "record_quote_load", lambda *_, **__: None)
    monkeypatch.setattr(cache_module, "max_quote_workers", 1)

    monkeypatch.setattr(iol_client_module.IOLClient, "_ensure_market_auth", lambda self: None)

    class StaticAuth:
        def __init__(self) -> None:
            self.tokens = {"access_token": "token", "refresh_token": "refresh"}

        def auth_header(self) -> dict[str, str]:
            return {"Authorization": "Bearer token"}

        def refresh(self) -> None:  # pragma: no cover - defensive
            raise AssertionError("refresh should not run")

    client = iol_client_module.IOLClient("user", "", auth=StaticAuth())

    def failing_request(self, method: str, url: str, **kwargs):  # type: ignore[no-untyped-def]
        raise _http_error(500)

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", failing_request)

    class Legacy429Stub:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_quote(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise _http_error(429)

    monkeypatch.setattr("infrastructure.iol.legacy.iol_client.IOLClient", Legacy429Stub)

    class OHLCStub:
        def __init__(self, *args, **kwargs) -> None:
            self._cache: dict[str, SimpleNamespace] = {}

        def _make_cache_key(self, symbol, params):  # type: ignore[no-untyped-def]
            return f"{symbol}|{params.get('period')}|{params.get('interval')}"

        def fetch(self, symbol, **params):  # type: ignore[no-untyped-def]
            key = self._make_cache_key(symbol, params)
            self._cache[key] = SimpleNamespace(provider="alpha_vantage")
            index = pd.to_datetime(["2024-01-01", "2024-01-02"])
            return pd.DataFrame({"Close": [100.0, 104.0]}, index=index)

    monkeypatch.setattr("services.ohlc_adapter.OHLCAdapter", OHLCStub)

    portfolio_payload = {
        "activos": [
            {"simbolo": "GGAL", "mercado": "bcba", "cantidad": 10, "costoUnitario": 100.0},
            {"simbolo": "AAPL", "mercado": "nyse", "cantidad": 5, "costoUnitario": 150.0},
        ]
    }

    psvc = PortfolioService()
    df_pos = psvc.normalize_positions(portfolio_payload)
    controls = Controls(hide_cash=False)

    try:
        df_view = filters.apply_filters(df_pos, controls, client, psvc)
    finally:
        cache_module._QUOTE_CACHE.clear()
        cache_module.fetch_quotes_bulk.clear()

    assert not df_view.empty
    assert (df_view["chg_%"].dropna() > 0).all()
