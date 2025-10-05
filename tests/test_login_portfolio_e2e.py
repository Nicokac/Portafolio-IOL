from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from application import auth_service
from application.portfolio_service import PortfolioService
from controllers.portfolio import filters, load_data
from domain.models import Controls
from services import cache as services_cache


class FakeStreamlit:
    """Minimal streamlit replacement used throughout the test."""

    @dataclass
    class _Spinner:
        owner: "FakeStreamlit"
        message: str

        def __enter__(self):  # pragma: no cover - trivial
            self.owner._spinner_messages.append(self.message)
            return self

        def __exit__(self, exc_type, exc, tb):  # pragma: no cover - trivial
            return False

    def __init__(self) -> None:
        self.session_state: dict = {}
        self._warnings: list[str] = []
        self._errors: list[str] = []
        self._infos: list[str] = []
        self._spinner_messages: list[str] = []
        self._rerun_called = False

    # --- Helpers mimicking streamlit API used by the application ---
    def spinner(self, message: str) -> "FakeStreamlit._Spinner":
        return FakeStreamlit._Spinner(self, message)

    def warning(self, message: str) -> None:
        self._warnings.append(message)

    def error(self, message: str) -> None:  # pragma: no cover - defensive
        self._errors.append(message)

    def info(self, message: str) -> None:  # pragma: no cover - defensive
        self._infos.append(message)

    def dataframe(self, *args, **kwargs) -> None:  # pragma: no cover - noop
        return None

    def caption(self, *args, **kwargs) -> None:  # pragma: no cover - noop
        return None

    def rerun(self) -> None:  # pragma: no cover - defensive
        self._rerun_called = True

    def stop(self) -> None:  # pragma: no cover - defensive
        raise RuntimeError("streamlit.stop() should not be invoked in tests")


@pytest.fixture
def fake_streamlit(monkeypatch) -> FakeStreamlit:
    fake_st = FakeStreamlit()
    for module in (auth_service, load_data, filters, services_cache):
        monkeypatch.setattr(module, "st", fake_st)
    return fake_st


class FakeClient:
    def __init__(self, portfolio_payload: dict, quotes: dict[tuple[str, str], dict]):
        self._payload = portfolio_payload
        self.quotes = quotes
        self.auth = SimpleNamespace(tokens_path="tokens/fake.json")
        self.portfolio_calls: int = 0
        self.quote_requests: list[tuple[tuple[str, str], ...]] = []

    def get_portfolio(self, country: str = "argentina") -> dict:
        self.portfolio_calls += 1
        return self._payload

    def get_quotes_bulk(self, items):
        self.quote_requests.append(tuple(items))
        out: dict[tuple[str, str], dict] = {}
        for mercado, simbolo in items:
            key = (str(mercado).lower(), str(simbolo).upper())
            out[key] = self.quotes.get(key, {"last": None, "chg_pct": None})
        return out


def test_login_portfolio_e2e(monkeypatch, fake_streamlit):
    """End-to-end flow covering login, portfolio normalization and filters."""

    portfolio_payload = {
        "activos": [
            {"simbolo": "AAPL", "mercado": "nyse", "cantidad": 10, "costoUnitario": 150},
            {"simbolo": "AL30", "mercado": "bcba", "cantidad": 20, "costoUnitario": 95},
            {"simbolo": "IOLPORA", "mercado": "bcba", "cantidad": 1000, "costoUnitario": 1.0},
        ]
    }
    quotes_map = {
        ("nyse", "AAPL"): {"last": 170.0, "chg_pct": 1.5, "cierreAnterior": 168.0},
        ("bcba", "AL30"): {"last": 98.0, "chg_pct": -0.5, "cierreAnterior": 98.5},
    }
    fake_client = FakeClient(portfolio_payload, quotes_map)

    def fake_fetch_portfolio(cli):
        assert cli is fake_client
        return cli.get_portfolio()

    def fake_fetch_quotes(cli, items):
        assert cli is fake_client
        return cli.get_quotes_bulk(items)

    monkeypatch.setattr(load_data, "fetch_portfolio", fake_fetch_portfolio)
    monkeypatch.setattr(filters, "fetch_quotes_bulk", fake_fetch_quotes)
    monkeypatch.setattr(services_cache, "fetch_portfolio", fake_fetch_portfolio)
    monkeypatch.setattr(services_cache, "fetch_quotes_bulk", fake_fetch_quotes)
    monkeypatch.setattr(services_cache, "fetch_fx_rates", lambda: ({"USDARS": 100.0}, None))

    class FakeAuthProvider:
        def __init__(self, client):
            self.client = client
            self.last_login = None

        def login(self, user: str, password: str) -> dict:
            self.last_login = (user, password)
            fake_streamlit.session_state["IOL_USERNAME"] = user
            return {"access_token": "token-abc", "refresh_token": "token-refresh"}

        def logout(self, user: str, password: str = "") -> None:  # pragma: no cover - defensive
            fake_streamlit.session_state.pop("IOL_USERNAME", None)

        def build_client(self):
            return self.client, None

    original_provider = auth_service.get_auth_provider()
    auth_service.register_auth_provider(FakeAuthProvider(fake_client))

    try:
        tokens = auth_service.login("alice", "wonderland")
        assert tokens == {"access_token": "token-abc", "refresh_token": "token-refresh"}
        assert "client_salt" in fake_streamlit.session_state

        cli, err = auth_service.get_auth_provider().build_client()
        assert err is None
        assert cli is fake_client
        assert fake_client.portfolio_calls == 0

        psvc = PortfolioService()
        df_pos, all_symbols, available_types = load_data.load_portfolio_data(cli, psvc)

        assert fake_client.portfolio_calls == 1
        assert set(all_symbols) == {"AAPL", "AL30", "IOLPORA"}
        assert {"CEDEAR", "Bono"}.issubset(set(available_types))

        controls = Controls(
            hide_cash=True,
            selected_syms=["AAPL"],
            selected_types=["CEDEAR"],
            symbol_query="AAP",
        )

        df_view = filters.apply_filters(df_pos, controls, cli, psvc)

        assert list(df_view["simbolo"]) == ["AAPL"]
        row = df_view.iloc[0]
        assert row["mercado"] == "NYSE"
        assert row["valor_actual"] == pytest.approx(1700.0)
        assert row["costo"] == pytest.approx(1500.0)
        assert row["pl"] == pytest.approx(200.0)
        assert row["pl_%"] == pytest.approx(13.333333, rel=1e-3)
        assert row["chg_%"] == pytest.approx(1.5)
        assert row["tipo"] == "CEDEAR"
        assert ("nyse", "AAPL") in fake_client.quote_requests[0]
        assert fake_streamlit.session_state["quotes_hist"]["AAPL"][0]["chg_pct"] == pytest.approx(1.5)
    finally:
        auth_service.register_auth_provider(original_provider)

