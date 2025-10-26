import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from application.portfolio_service import PortfolioService
from controllers.portfolio import filters
from domain.models import Controls
from infrastructure.iol import client as iol_client_module
from services import cache as cache_module
from services.portfolio_view import PortfolioViewModelService
from tests.fixtures.streamlit import BaseFakeStreamlit
from tests.ui.test_portfolio_ui import FakeStreamlit

pytestmark = [
    pytest.mark.usefixtures("restore_real_iol_client"),
    pytest.mark.parametrize("fake_st", ["base"], indirect=True),
]

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _http_error(status: int) -> requests.HTTPError:
    response = SimpleNamespace(status_code=status)
    return requests.HTTPError(response=response)


@pytest.fixture
def fake_streamlit(monkeypatch: pytest.MonkeyPatch, fake_st: BaseFakeStreamlit) -> BaseFakeStreamlit:
    for module in (filters, cache_module, iol_client_module):
        monkeypatch.setattr(module, "st", fake_st)
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

    monkeypatch.setattr("infrastructure.iol.compat.iol_client.IOLClient", Legacy429Stub)

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
            {
                "simbolo": "GGAL",
                "mercado": "bcba",
                "cantidad": 10,
                "costoUnitario": 100.0,
            },
            {
                "simbolo": "AAPL",
                "mercado": "nyse",
                "cantidad": 5,
                "costoUnitario": 150.0,
            },
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


def test_portfolio_country_endpoint_flow(
    monkeypatch: pytest.MonkeyPatch,
    fake_streamlit: FakeStreamlit,
    tmp_path,
) -> None:
    cache_file = tmp_path / "portfolio.json"
    monkeypatch.setattr("infrastructure.iol.client.PORTFOLIO_CACHE", cache_file)
    monkeypatch.setattr(
        iol_client_module.IOLClient,
        "_ensure_market_auth",
        lambda self: None,
        raising=False,
    )

    class DummyAuth:
        def __init__(self) -> None:
            self.tokens = {"access_token": "tok", "refresh_token": "ref"}

        def auth_header(self) -> dict[str, str]:
            return {"Authorization": "Bearer tok"}

        def refresh(self) -> None:  # pragma: no cover - defensive
            raise AssertionError("refresh should not run")

    payload = {
        "pais": "argentina",
        "activos": [
            {
                "simbolo": "GGAL",
                "mercado": "bcba",
                "cantidad": 10,
                "costoUnitario": 100.0,
                "valorizado": 1250.0,
                "titulo": {"tipo": "Acción"},
            },
            {
                "simbolo": "AAPL",
                "mercado": "nyse",
                "cantidad": 5,
                "costoUnitario": 150.0,
                "valorizado": 850.0,
                "titulo": {"tipo": "CEDEAR"},
            },
        ],
    }

    class DummyResponse:
        def __init__(self, data: dict) -> None:
            self._data = data
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._data

    captured: dict[str, str] = {}

    def fake_request(self, method: str, url: str, **kwargs):  # type: ignore[no-untyped-def]
        captured["method"] = method
        captured["url"] = url
        return DummyResponse(payload)

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", fake_request, raising=False)

    client = iol_client_module.IOLClient("user", "", auth=DummyAuth())
    data = client.get_portfolio()

    assert captured["method"] == "GET"
    assert captured["url"].endswith("/portafolio/argentina")
    assert data == payload
    assert json.loads(cache_file.read_text()) == payload

    from controllers.portfolio import load_data as load_data_module

    class _Spinner:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    fake_streamlit.spinner = lambda message: _Spinner()
    fake_streamlit.info = lambda *a, **k: None
    fake_streamlit.dataframe = lambda *a, **k: None

    monkeypatch.setattr(load_data_module, "st", fake_streamlit)
    monkeypatch.setattr(load_data_module, "fetch_portfolio", lambda cli: data)

    psvc = PortfolioService()
    df_pos, all_syms, available_types = load_data_module.load_portfolio_data(client, psvc)

    assert not df_pos.empty
    assert set(all_syms) == {"AAPL", "GGAL"}
    assert available_types  # clasificación derivada del payload

    assets = {
        (
            str(item.get("mercado", "")).lower(),
            str(item.get("simbolo", "")).upper(),
        ): item
        for item in payload["activos"]
    }

    def fake_quotes(_cli, pairs):
        out: dict[tuple[str, str], dict[str, float]] = {}
        for mercado, simbolo in pairs:
            key = (str(mercado).lower(), str(simbolo).upper())
            asset = assets.get(key, {})
            qty = float(asset.get("cantidad") or 0.0)
            valorizado = float(asset.get("valorizado") or 0.0)
            last = valorizado / qty if qty else 0.0
            out[key] = {"last": last, "chg_pct": 0.0}
        return out

    monkeypatch.setattr(filters, "fetch_quotes_bulk", fake_quotes)
    monkeypatch.setattr(cache_module, "fetch_quotes_bulk", fake_quotes)

    view_service = PortfolioViewModelService()
    controls = Controls(hide_cash=False)
    snapshot = view_service.get_portfolio_view(df_pos, controls, cli=client, psvc=psvc)

    total_valorizado = sum(item["valorizado"] for item in payload["activos"])
    assert snapshot.totals.total_value == pytest.approx(total_valorizado)
    assert (snapshot.df_view["valor_actual"] > 0).all()
