"""Integration test that exercises the opportunities tab end-to-end."""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import pytest
from streamlit.runtime.secrets import Secrets
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_YAHOO_TEST_MODULE = PROJECT_ROOT / "tests" / "application" / "test_screener_yahoo.py"
_YAHOO_SPEC = importlib.util.spec_from_file_location(
    "tests.application.test_screener_yahoo",
    _YAHOO_TEST_MODULE,
)
assert _YAHOO_SPEC and _YAHOO_SPEC.loader is not None
_YAHOO_MODULE = importlib.util.module_from_spec(_YAHOO_SPEC)
_YAHOO_SPEC.loader.exec_module(_YAHOO_MODULE)
build_bulk_fake_yahoo_client = _YAHOO_MODULE.build_bulk_fake_yahoo_client


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
    app.run(timeout=10)
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


def test_opportunities_flow_applies_critical_filters_with_stub_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from application.screener import opportunities as opportunities_module
    import shared.config as shared_config
    import shared.settings as shared_settings
    import controllers.opportunities as controller_module

    synthetic_base = [
        {
            "ticker": "ALFA",
            "payout_ratio": 3.0,
            "dividend_streak": 45,
            "cagr": 24.0,
            "dividend_yield": 1.8,
            "price": 210.0,
            "rsi": 50.0,
            "sma_50": 205.0,
            "sma_200": 198.0,
            "market_cap": 95_000.0,
            "pe_ratio": 18.0,
            "revenue_growth": 24.0,
            "is_latam": False,
            "trailing_eps": 5.0,
            "forward_eps": 6.5,
            "buyback": 18.0,
            "sector": "Technology",
        },
        {
            "ticker": "BETA",
            "payout_ratio": 6.0,
            "dividend_streak": 38,
            "cagr": 23.0,
            "dividend_yield": 1.5,
            "price": 198.0,
            "rsi": 49.0,
            "sma_50": 192.0,
            "sma_200": 183.0,
            "market_cap": 84_000.0,
            "pe_ratio": 20.0,
            "revenue_growth": 22.0,
            "is_latam": False,
            "trailing_eps": 4.5,
            "forward_eps": 5.7,
            "buyback": 16.0,
            "sector": "Technology",
        },
        {
            "ticker": "GAMA",
            "payout_ratio": 8.0,
            "dividend_streak": 35,
            "cagr": 22.0,
            "dividend_yield": 1.2,
            "price": 176.0,
            "rsi": 51.0,
            "sma_50": 170.0,
            "sma_200": 160.0,
            "market_cap": 78_000.0,
            "pe_ratio": 21.0,
            "revenue_growth": 21.0,
            "is_latam": False,
            "trailing_eps": 4.2,
            "forward_eps": 5.2,
            "buyback": 14.0,
            "sector": "Technology",
        },
        {
            "ticker": "DELTA",
            "payout_ratio": 10.0,
            "dividend_streak": 33,
            "cagr": 21.0,
            "dividend_yield": 1.1,
            "price": 162.0,
            "rsi": 48.0,
            "sma_50": 158.0,
            "sma_200": 150.0,
            "market_cap": 72_000.0,
            "pe_ratio": 21.0,
            "revenue_growth": 20.0,
            "is_latam": False,
            "trailing_eps": 4.0,
            "forward_eps": 5.0,
            "buyback": 13.0,
            "sector": "Technology",
        },
        {
            "ticker": "LATM",
            "payout_ratio": 32.0,
            "dividend_streak": 28,
            "cagr": 18.0,
            "dividend_yield": 2.4,
            "price": 118.0,
            "rsi": 55.0,
            "sma_50": 112.0,
            "sma_200": 104.0,
            "market_cap": 45_000.0,
            "pe_ratio": 19.5,
            "revenue_growth": 19.0,
            "is_latam": True,
            "trailing_eps": 3.9,
            "forward_eps": 4.2,
            "buyback": 7.5,
            "sector": "Technology",
        },
        {
            "ticker": "VALUE",
            "payout_ratio": 42.0,
            "dividend_streak": 18,
            "cagr": 10.0,
            "dividend_yield": 3.1,
            "price": 88.0,
            "rsi": 60.0,
            "sma_50": 84.0,
            "sma_200": 76.0,
            "market_cap": 15_000.0,
            "pe_ratio": 28.0,
            "revenue_growth": 8.0,
            "is_latam": False,
            "trailing_eps": 3.8,
            "forward_eps": 4.1,
            "buyback": 2.0,
            "sector": "Industrials",
        },
    ]

    monkeypatch.setattr(
        opportunities_module,
        "_BASE_OPPORTUNITIES",
        synthetic_base,
        raising=False,
    )

    monkeypatch.setattr(shared_settings, "max_results", 3, raising=False)
    monkeypatch.setattr(shared_settings, "MAX_RESULTS", 3, raising=False)
    monkeypatch.setattr(shared_config.settings, "max_results", 3, raising=False)

    def _stubbed_generate(filters: Optional[Mapping[str, object]] = None) -> Mapping[str, object]:
        filters = dict(filters or {})
        result = opportunities_module.run_screener_stub(
            manual_tickers=filters.get("manual_tickers") or filters.get("tickers"),
            exclude_tickers=filters.get("exclude_tickers"),
            max_payout=filters.get("max_payout"),
            min_div_streak=filters.get("min_div_streak"),
            min_cagr=filters.get("min_cagr"),
            sectors=filters.get("sectors"),
            include_technicals=bool(filters.get("include_technicals", False)),
            min_market_cap=filters.get("min_market_cap"),
            max_pe=filters.get("max_pe"),
            min_revenue_growth=filters.get("min_revenue_growth"),
            include_latam=filters.get("include_latam", True),
            min_eps_growth=filters.get("min_eps_growth"),
            min_buyback=filters.get("min_buyback"),
            min_score_threshold=filters.get("min_score_threshold"),
            max_results=filters.get("max_results"),
        )
        if isinstance(result, tuple):
            table, notes = result
        else:
            table, notes = result, []
        return {"table": table, "notes": notes, "source": "stub"}

    monkeypatch.setattr(
        controller_module, "generate_opportunities_report", _stubbed_generate
    )

    app = _render_app()

    _set_number_input(app, "Capitalización mínima (US$ MM)", 20_000)
    _set_number_input(app, "P/E máximo", 22.0)
    _set_number_input(app, "Crecimiento ingresos mínimo (%)", 20.0)
    _set_number_input(app, "Payout máximo (%)", 15.0)
    _set_slider(app, "Racha mínima de dividendos (años)", 30)
    _set_number_input(app, "CAGR mínimo de dividendos (%)", 20.0)
    _set_number_input(app, "Crecimiento mínimo de EPS (%)", 15.0)
    _set_number_input(app, "Buyback mínimo (%)", 8.0)
    _set_checkbox(app, "Incluir Latam", False)
    _set_slider(app, "Score mínimo", 80)
    _set_number_input(app, "Máximo de resultados", 3)
    _set_multiselect(app, "Sectores", ["Technology"])

    app.run()

    search_buttons = [
        element
        for element in app.get("button")
        if getattr(element, "key", None) == "search_opportunities"
    ]
    assert search_buttons, "Expected the opportunities search button to be present"

    search_buttons[0].click()

    start = time.perf_counter()
    app.run()
    elapsed = time.perf_counter() - start
    assert elapsed <= 5.0, f"Screening exceeded time budget: {elapsed:.2f}s"

    dataframes = app.get("arrow_data_frame")
    assert dataframes, "Expected the filtered results dataframe to be rendered"
    displayed = dataframes[0].value
    assert isinstance(displayed, pd.DataFrame)
    assert len(displayed) == 3
    assert list(displayed["ticker"]) == ["ALFA", "BETA", "GAMA"]
    scores = pd.to_numeric(displayed["score_compuesto"], errors="coerce")
    assert not scores.isna().any()
    assert (scores >= 80).all()

    markdown_blocks = [element.value for element in app.get("markdown")]
    notes = [block for block in markdown_blocks if "Se muestran" in block]
    assert notes, "Expected truncation warning to be present in notes"
    assert any("máximo solicitado" in note for note in notes)

    captions = [element.value for element in app.get("caption")]
    assert any("Resultados simulados" in caption for caption in captions)


def test_yahoo_large_universe_e2e(monkeypatch: pytest.MonkeyPatch) -> None:
    bulk_client = build_bulk_fake_yahoo_client()
    listings = bulk_client.list_symbols_by_markets(["BULK"])
    assert len(listings) >= 500

    monkeypatch.setattr(
        "application.screener.opportunities._get_target_markets",
        lambda: ["BULK"],
    )
    monkeypatch.setattr(
        "application.screener.opportunities.YahooFinanceClient",
        lambda: bulk_client,
    )

    app = _render_app()

    _set_number_input(app, "Capitalización mínima (US$ MM)", 0)
    _set_number_input(app, "P/E máximo", 40.0)
    _set_number_input(app, "Crecimiento ingresos mínimo (%)", 0.0)
    _set_number_input(app, "Payout máximo (%)", 80.0)
    _set_slider(app, "Racha mínima de dividendos (años)", 0)
    _set_number_input(app, "CAGR mínimo de dividendos (%)", 0.0)
    _set_number_input(app, "Crecimiento mínimo de EPS (%)", 0.0)
    _set_number_input(app, "Buyback mínimo (%)", 0.0)
    _set_checkbox(app, "Incluir Latam", False)
    _set_slider(app, "Score mínimo", 0)
    _set_number_input(app, "Máximo de resultados", 10)
    _set_multiselect(
        app,
        "Sectores",
        ["Technology", "Healthcare", "Industrials"],
    )

    app.run()

    search_buttons = [
        element
        for element in app.get("button")
        if getattr(element, "key", None) == "search_opportunities"
    ]
    assert search_buttons, "Expected search button to be present"

    start = time.perf_counter()
    search_buttons[0].click()
    app.run(timeout=10)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"Execution took too long: {elapsed:.2f} seconds"

    dataframes = app.get("arrow_data_frame")
    assert dataframes, "Expected the opportunities dataframe to be rendered"
    displayed = dataframes[0].value
    assert isinstance(displayed, pd.DataFrame)
    assert len(displayed) == 10, "Expected the table to be truncated to 10 rows"
    assert displayed["ticker"].str.startswith("BULK").all()
    scores = pd.to_numeric(displayed["score_compuesto"], errors="coerce").dropna()
    assert len(scores) == len(displayed)
    assert list(scores) == sorted(scores, reverse=True)

    markdown_blocks = [element.value for element in app.get("markdown")]
    truncation_notes = [
        block
        for block in markdown_blocks
        if block.startswith("-") and "máximo solicitado" in block.lower()
    ]
    assert (
        truncation_notes
    ), "Expected a note indicating the dataset was truncated to the requested maximum"


def test_yahoo_large_universe_emits_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    bulk_client = build_bulk_fake_yahoo_client()
    monkeypatch.setattr(
        "application.screener.opportunities._get_target_markets",
        lambda: ["BULK"],
    )
    monkeypatch.setattr(
        "application.screener.opportunities.YahooFinanceClient",
        lambda: bulk_client,
    )

    class _TelemetryLogger:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def info(self, message: str, *args, **kwargs) -> None:  # pragma: no cover - formatting helper
            formatted = message % args if args else message
            self.messages.append(formatted)

        def __getattr__(self, name: str):  # pragma: no cover - passthrough for unused levels
            def _noop(*_args, **_kwargs) -> None:
                return None

            return _noop

    telemetry_logger = _TelemetryLogger()
    monkeypatch.setattr(
        "application.screener.opportunities.LOGGER", telemetry_logger
    )

    app = _render_app()

    _set_number_input(app, "Capitalización mínima (US$ MM)", 0)
    _set_number_input(app, "P/E máximo", 40.0)
    _set_number_input(app, "Crecimiento ingresos mínimo (%)", 0.0)
    _set_number_input(app, "Payout máximo (%)", 80.0)
    _set_slider(app, "Racha mínima de dividendos (años)", 0)
    _set_number_input(app, "CAGR mínimo de dividendos (%)", 0.0)
    _set_number_input(app, "Crecimiento mínimo de EPS (%)", 0.0)
    _set_number_input(app, "Buyback mínimo (%)", 0.0)
    _set_checkbox(app, "Incluir Latam", False)
    _set_slider(app, "Score mínimo", 0)
    _set_number_input(app, "Máximo de resultados", 10)
    _set_multiselect(
        app,
        "Sectores",
        ["Technology", "Healthcare", "Industrials"],
    )

    app.run()

    search_buttons = [
        element
        for element in app.get("button")
        if getattr(element, "key", None) == "search_opportunities"
    ]
    assert search_buttons, "Expected search button to be present"
    search_buttons[0].click()

    app.run(timeout=10)

    assert telemetry_logger.messages, "Expected telemetry info log to be emitted"
    assert any(
        "Yahoo screener processed" in message
        for message in telemetry_logger.messages
    ), "Expected Yahoo telemetry log entry"

    markdown_blocks = [element.value for element in app.get("markdown")]
    assert any(
        "Yahoo procesó" in block for block in markdown_blocks
    ), "Expected telemetry note to be propagated to the UI"
