"""Integration test that exercises the opportunities tab end-to-end."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import pytest
from streamlit.runtime.secrets import Secrets
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_streamlit_module():
    import streamlit as streamlit_module

    if getattr(streamlit_module, "__file__", None) and hasattr(
        streamlit_module, "button"
    ):
        return streamlit_module

    for name in list(sys.modules):
        if name == "streamlit" or name.startswith("streamlit."):
            sys.modules.pop(name, None)

    import streamlit as real_streamlit

    return real_streamlit


st = _resolve_streamlit_module()
sys.modules["streamlit"] = st


@pytest.fixture(autouse=True)
def _enable_opportunities_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    import shared.config as shared_config
    import shared.settings as shared_settings

    monkeypatch.setattr(shared_settings, "FEATURE_OPPORTUNITIES_TAB", True)
    monkeypatch.setenv("FEATURE_OPPORTUNITIES_TAB", "true")
    monkeypatch.setattr(shared_config.settings, "tokens_key", "dummy", raising=False)
    monkeypatch.setattr(
        shared_config.settings, "allow_plain_tokens", True, raising=False
    )


class _FakeYahooClient:
    """Deterministic Yahoo client used to avoid external requests."""

    def __init__(self) -> None:
        self.market_calls: list[tuple[str, ...]] = []
        self._fundamentals: Mapping[str, Mapping[str, object]] = {
            "SAFE": {
                "payout_ratio": 35.0,
                "dividend_yield": 1.2,
                "market_cap": 1_200.0,
                "pe_ratio": 18.0,
                "revenue_growth": 15.0,
                "country": "United States",
                "sector": "Technology",
                "trailing_eps": 6.2,
                "forward_eps": 6.8,
            },
            "SPEC": {
                "payout_ratio": 82.0,
                "dividend_yield": 2.1,
                "market_cap": 640.0,
                "pe_ratio": 27.5,
                "revenue_growth": 6.0,
                "country": "Brazil",
                "sector": "Utilities",
                "trailing_eps": 3.4,
                "forward_eps": 3.6,
            },
        }

    def list_symbols_by_markets(self, markets: Iterable[str]) -> list[Mapping[str, object]]:
        self.market_calls.append(tuple(markets))
        return [
            {
                "ticker": "SAFE",
                "market_cap": 1_200.0,
                "pe_ratio": 18.0,
                "revenue_growth": 15.0,
                "country": "United States",
                "sector": "Technology",
            },
            {
                "ticker": "SPEC",
                "market_cap": 640.0,
                "pe_ratio": 27.5,
                "revenue_growth": 6.0,
                "country": "Brazil",
                "sector": "Utilities",
            },
        ]

    def get_fundamentals(self, ticker: str) -> Mapping[str, object]:
        return dict(self._fundamentals[ticker])

    def get_dividends(self, ticker: str) -> pd.DataFrame:
        years = range(2018, 2024)
        if ticker == "SAFE":
            amounts = np.linspace(1.6, 1.1, num=len(years))
        else:
            years = range(2020, 2023)
            amounts = np.linspace(0.6, 0.9, num=len(years))
        data = {
            "date": [pd.Timestamp(year=year, month=3, day=1) for year in years],
            "amount": amounts,
        }
        return pd.DataFrame(data)

    def get_shares_outstanding(self, ticker: str) -> pd.DataFrame:
        if ticker == "SAFE":
            shares = [120_000_000, 115_000_000, 110_000_000]
        else:
            shares = [150_000_000, 152_000_000, 155_000_000]
        dates = pd.date_range("2021-01-01", periods=len(shares), freq="YS")
        return pd.DataFrame({"date": dates, "shares": shares})

    def get_price_history(self, ticker: str) -> pd.DataFrame:
        base_price = 100.0 if ticker == "SAFE" else 45.0
        dates = pd.date_range("2021-01-01", periods=260, freq="B")
        trend = np.linspace(base_price, base_price + 50.0, num=len(dates))
        frame = pd.DataFrame({"date": dates, "close": trend, "adj_close": trend})
        return frame


class _HighScoreYahooClient:
    """Yahoo client tailored to exercise advanced filters simultaneously."""

    def __init__(self) -> None:
        self.market_calls: list[tuple[str, ...]] = []
        self._fundamentals: Mapping[str, Mapping[str, object]] = {
            "ELITE": {
                "payout_ratio": 0.0,
                "dividend_yield": 1.8,
                "market_cap": 1_800.0,
                "pe_ratio": 18.0,
                "revenue_growth": 24.0,
                "country": "United States",
                "sector": "Technology",
                "trailing_eps": 4.0,
                "forward_eps": 5.6,
            },
            "ALPHA": {
                "payout_ratio": 2.0,
                "dividend_yield": 1.5,
                "market_cap": 1_650.0,
                "pe_ratio": 19.0,
                "revenue_growth": 20.0,
                "country": "United States",
                "sector": "Technology",
                "trailing_eps": 3.5,
                "forward_eps": 4.6,
            },
            "LAGG": {
                "payout_ratio": 55.0,
                "dividend_yield": 0.8,
                "market_cap": 1_200.0,
                "pe_ratio": 24.0,
                "revenue_growth": 5.0,
                "country": "United States",
                "sector": "Technology",
                "trailing_eps": 2.0,
                "forward_eps": 2.1,
            },
        }

    def list_symbols_by_markets(self, markets: Iterable[str]) -> list[Mapping[str, object]]:
        self.market_calls.append(tuple(markets))
        return [
            {
                "ticker": "ELITE",
                "market_cap": 1_800.0,
                "pe_ratio": 18.0,
                "revenue_growth": 24.0,
                "country": "United States",
                "sector": "Technology",
            },
            {
                "ticker": "ALPHA",
                "market_cap": 1_650.0,
                "pe_ratio": 19.0,
                "revenue_growth": 20.0,
                "country": "United States",
                "sector": "Technology",
            },
            {
                "ticker": "LAGG",
                "market_cap": 1_200.0,
                "pe_ratio": 24.0,
                "revenue_growth": 5.0,
                "country": "United States",
                "sector": "Technology",
            },
        ]

    def get_fundamentals(self, ticker: str) -> Mapping[str, object]:
        return dict(self._fundamentals[ticker])

    def get_dividends(self, ticker: str) -> pd.DataFrame:  # pragma: no cover - exercised via AppTest
        # Returning an empty frame keeps the dividend streak filter neutral while
        # still exercising the Yahoo controller code paths.
        return pd.DataFrame(columns=["date", "amount"])

    def get_shares_outstanding(self, ticker: str) -> pd.DataFrame:
        if ticker == "ELITE":
            shares = [220_000_000, 200_000_000, 176_000_000]
        elif ticker == "ALPHA":
            shares = [180_000_000, 168_000_000, 150_000_000]
        else:
            shares = [120_000_000, 122_000_000, 125_000_000]
        dates = pd.date_range("2021-01-01", periods=len(shares), freq="YS")
        return pd.DataFrame({"date": dates, "shares": shares})

    def get_price_history(self, ticker: str) -> pd.DataFrame:
        dates = pd.date_range("2023-01-01", periods=12, freq="B")
        if ticker == "ELITE":
            base = 103.0
        elif ticker == "ALPHA":
            base = 93.0
        else:
            base = 71.0
        trend = np.linspace(0, 3, num=len(dates))
        close = base + trend if ticker != "LAGG" else base - trend
        return pd.DataFrame({"date": dates, "close": close, "adj_close": close})


def _render_app() -> AppTest:
    if not hasattr(st, "secrets"):
        st.secrets = Secrets({})
    script = "\n".join(
        [
            "import sys",
            f"sys.path.insert(0, {repr(str(PROJECT_ROOT))})",
            "from ui.tabs.opportunities import render_opportunities_tab",
            "render_opportunities_tab()",
        ]
    )
    app = AppTest.from_string(script)
    app.run()
    return app


def _set_number_input(app: AppTest, label: str, value: float | int) -> None:
    widget = next(element for element in app.get("number_input") if element.label == label)
    widget.set_value(value)


def _set_slider(app: AppTest, label: str, value: float | int) -> None:
    widget = next(element for element in app.get("slider") if element.label == label)
    widget.set_value(value)


def _set_checkbox(app: AppTest, label: str, value: bool) -> None:
    widget = next(element for element in app.get("checkbox") if element.label == label)
    widget.set_value(value)


def _set_multiselect(app: AppTest, label: str, value: Iterable[str]) -> None:
    widget = next(
        element for element in app.get("multiselect") if element.label == label
    )
    widget.set_value(list(value))


def test_opportunities_flow_renders_yahoo_results(monkeypatch: pytest.MonkeyPatch) -> None:
    created_clients: list[_FakeYahooClient] = []

    def _factory() -> _FakeYahooClient:
        client = _FakeYahooClient()
        created_clients.append(client)
        return client

    monkeypatch.setattr(
        "application.screener.opportunities.YahooFinanceClient", _factory
    )

    app = _render_app()

    _set_number_input(app, "Capitalización mínima (US$ MM)", 750)
    _set_number_input(app, "P/E máximo", 22.0)
    _set_number_input(app, "Crecimiento ingresos mínimo (%)", 10.0)
    _set_number_input(app, "Payout máximo (%)", 50.0)
    _set_slider(app, "Racha mínima de dividendos (años)", 5)
    _set_number_input(app, "CAGR mínimo de dividendos (%)", 4.0)
    _set_number_input(app, "Buyback mínimo (%)", 0.5)
    _set_checkbox(app, "Incluir Latam", False)
    _set_slider(app, "Score mínimo", 0)
    _set_number_input(app, "Máximo de resultados", 5)
    _set_multiselect(app, "Sectores", ["Technology"])

    app.run()

    buttons = [
        element
        for element in app.get("button")
        if getattr(element, "key", None) == "search_opportunities"
    ]
    assert buttons, "Expected to find the opportunities search button"
    buttons[0].click()

    app.run()

    assert created_clients, "Expected YahooFinanceClient to be instantiated"
    fake_client = created_clients[0]
    assert fake_client.market_calls, "Expected markets to be requested from Yahoo client"

    dataframes = app.get("arrow_data_frame")
    assert dataframes, "Expected the results dataframe to be rendered"
    displayed = dataframes[0].value
    assert isinstance(displayed, pd.DataFrame)
    assert list(displayed["ticker"]) == ["SAFE"]

    captions = [element.value for element in app.get("caption")]
    assert "Resultados obtenidos de Yahoo Finance" in captions
    assert any("ℹ️ Los filtros avanzados" in caption for caption in captions)

    markdown_blocks = [element.value for element in app.get("markdown")]
    assert "### Notas" in markdown_blocks
    bullet_points = [block for block in markdown_blocks if block.startswith("-")]
    assert any("📈 Analizando" in block for block in bullet_points)
    assert any("Filtros aplicados" in block for block in bullet_points)


def test_opportunities_flow_applies_growth_buyback_and_score_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_clients: list[_HighScoreYahooClient] = []

    def _factory() -> _HighScoreYahooClient:
        client = _HighScoreYahooClient()
        created_clients.append(client)
        return client

    monkeypatch.setattr(
        "application.screener.opportunities.YahooFinanceClient", _factory
    )

    app = _render_app()

    _set_number_input(app, "Capitalización mínima (US$ MM)", 1_500)
    _set_number_input(app, "P/E máximo", 20.0)
    _set_number_input(app, "Crecimiento ingresos mínimo (%)", 18.0)
    _set_number_input(app, "Payout máximo (%)", 100.0)
    _set_slider(app, "Racha mínima de dividendos (años)", 0)
    _set_number_input(app, "CAGR mínimo de dividendos (%)", 0.0)
    _set_number_input(app, "Crecimiento mínimo de EPS (%)", 20.0)
    _set_number_input(app, "Buyback mínimo (%)", 5.0)
    _set_checkbox(app, "Incluir Latam", False)
    _set_slider(app, "Score mínimo", 80)
    _set_number_input(app, "Máximo de resultados", 1)
    _set_multiselect(app, "Sectores", ["Technology"])

    app.run()

    search_buttons = [
        element
        for element in app.get("button")
        if getattr(element, "key", None) == "search_opportunities"
    ]
    assert search_buttons, "Expected the opportunities search button to be present"
    search_buttons[0].click()

    app.run()

    assert created_clients, "Expected the high-score Yahoo client to be instantiated"
    fake_client = created_clients[0]
    assert fake_client.market_calls, "Expected markets to be requested from Yahoo client"

    dataframes = app.get("arrow_data_frame")
    assert dataframes, "Expected the filtered results dataframe to be rendered"
    displayed = dataframes[0].value
    assert isinstance(displayed, pd.DataFrame)
    assert len(displayed) == 1
    assert list(displayed["ticker"]) == ["ELITE"]
    score_values = pd.to_numeric(displayed["score_compuesto"], errors="coerce")
    assert not score_values.isna().any()
    assert (score_values >= 80).all()

    captions = [element.value for element in app.get("caption")]
    assert any("Resultados obtenidos de Yahoo Finance" in caption for caption in captions)

    markdown_blocks = [element.value for element in app.get("markdown")]
    assert "### Notas" in markdown_blocks
    bullet_points = [block for block in markdown_blocks if block.startswith("-")]
    assert any("Se muestran 1 resultados de 2" in block for block in bullet_points)
    assert any("Filtros aplicados" in block for block in bullet_points)
