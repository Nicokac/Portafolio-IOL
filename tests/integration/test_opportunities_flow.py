"""Integration test that exercises the opportunities tab end-to-end."""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import pytest
from streamlit.runtime.secrets import Secrets
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.ui import notes as shared_notes

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


_CRITICAL_SECTORS = (
    "Technology",
    "Energy",
    "Industrials",
    "Consumer",
    "Healthcare",
    "Financials",
    "Utilities",
    "Materials",
)


def _assert_distribution(dataset: Iterable[Mapping[str, object]]) -> None:
    frame = pd.DataFrame(dataset)
    counts = frame["sector"].value_counts()
    for sector in _CRITICAL_SECTORS:
        count = int(counts.get(sector, 0))
        assert (
            count >= 3
        ), f"El sector crítico '{sector}' debería contar con al menos 3 emisores en el stub"
        assert (
            count <= 5
        ), f"El sector crítico '{sector}' debería permanecer acotado para pruebas deterministas"


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
        years = range(2014, 2024)
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
        dates = pd.date_range("2016-01-01", periods=6 * 12, freq="MS")
        start = base_price * 0.6 if ticker == "SAFE" else base_price * 0.7
        end = base_price + 60.0 if ticker == "SAFE" else base_price + 25.0
        trend = np.linspace(start, end, num=len(dates))
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
        years = range(2018, 2024)
        if ticker == "ELITE":
            amounts = np.linspace(1.8, 1.4, num=len(years))
        elif ticker == "ALPHA":
            amounts = np.linspace(1.6, 1.2, num=len(years))
        else:
            amounts = np.linspace(0.7, 0.6, num=len(years))
        data = {
            "date": [pd.Timestamp(year=year, month=3, day=1) for year in years],
            "amount": amounts,
        }
        return pd.DataFrame(data)

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
        dates = pd.date_range("2018-01-01", periods=5 * 12, freq="MS")
        if ticker == "ELITE":
            start, end, amplitude = 60.0, 220.0, 40.0
        elif ticker == "ALPHA":
            start, end, amplitude = 55.0, 200.0, 35.0
        else:
            start, end, amplitude = 85.0, 70.0, 5.0
        trend = np.linspace(start, end, num=len(dates))
        oscillation = amplitude * np.sin(np.linspace(0, 6 * np.pi, len(dates)))
        close = trend + oscillation
        frame = pd.DataFrame({"date": dates, "close": close, "adj_close": close})
        return frame


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


def _set_selectbox(app: AppTest, label: str, value: str) -> None:
    widget = next(element for element in app.get("selectbox") if element.label == label)
    widget.set_value(value)


def _click_search_button(app: AppTest) -> None:
    buttons = [
        element
        for element in app.get("button")
        if getattr(element, "key", None) == "search_opportunities"
    ]
    assert buttons, "Expected the opportunities search button to be present"
    buttons[0].click()


def _execute_search_cycle(app: AppTest, *, timeout: float = 10.0) -> tuple[pd.DataFrame, list[str], list[str]]:
    _click_search_button(app)
    app.run(timeout=timeout)

    dataframes = app.get("arrow_data_frame")
    assert dataframes, "Expected the results dataframe to be rendered"
    displayed = dataframes[0].value
    assert isinstance(displayed, pd.DataFrame)

    captions = [element.value for element in app.get("caption")]
    markdown_blocks = [element.value for element in app.get("markdown")]
    return displayed, captions, markdown_blocks


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

    _click_search_button(app)
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
    assert any(
        "Los filtros avanzados" in caption for caption in captions
    ), "Expected advanced filters caption to be rendered"

    markdown_blocks = [element.value for element in app.get("markdown")]
    assert any(
        block.startswith("### Notas") for block in markdown_blocks
    ), "Expected notes heading to be rendered"
    assert any("Analizando" in block for block in markdown_blocks)
    assert any("Filtros aplicados" in block for block in markdown_blocks)


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
    _set_slider(app, "Score mínimo", 50)
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
    assert (score_values >= 50).all()


def test_opportunities_flow_applies_critical_filters_with_stub_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from application.screener import opportunities as opportunities_module
    import shared.config as shared_config
    import shared.settings as shared_settings
    import controllers.opportunities as controller_module

    _assert_distribution(opportunities_module._BASE_OPPORTUNITIES)

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
    assert any("Stub procesó" in block for block in markdown_blocks)

    captions = [element.value for element in app.get("caption")]
    assert any("Resultados simulados" in caption for caption in captions)


def test_fallback_stub_emits_runtime_telemetry_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from controllers import opportunities as controller_module
    import application.screener.opportunities as opportunities_module

    monkeypatch.setattr(controller_module, "run_screener_yahoo", None, raising=False)

    def _configure_perf_counter(values: Iterable[float]) -> None:
        sequence = list(values)
        calls = iter(sequence)

        def _fake_perf_counter() -> float:
            try:
                return next(calls)
            except StopIteration:
                return sequence[-1]

        monkeypatch.setattr(opportunities_module.time, "perf_counter", _fake_perf_counter)

    def _extract_stub_note(notes: Iterable[str]) -> str:
        for note in notes:
            if note.startswith("ℹ️ Stub procesó"):
                return note
        raise AssertionError("Se esperaba una nota del stub")

    _configure_perf_counter([10.0, 10.05])
    df_fast, notes_fast, source_fast = controller_module.run_opportunities_controller(
        manual_tickers=None,
        exclude_tickers=None,
        max_payout=None,
        min_div_streak=None,
        min_cagr=None,
        min_market_cap=None,
        max_pe=None,
        min_revenue_growth=None,
        include_latam=None,
        include_technicals=False,
        min_eps_growth=None,
        min_buyback=None,
        min_score_threshold=None,
        max_results=None,
        sectors=None,
    )

    fast_note = _extract_stub_note(notes_fast)
    assert source_fast == "stub"
    severity_fast, _, matched_fast = shared_notes.classify_note(fast_note)
    assert severity_fast == "info"
    assert matched_fast
    assert df_fast.attrs["_notes"][-1] == fast_note

    _configure_perf_counter([20.0, 20.5])
    df_slow, notes_slow, _ = controller_module.run_opportunities_controller(
        manual_tickers=None,
        exclude_tickers=None,
        max_payout=None,
        min_div_streak=None,
        min_cagr=None,
        min_market_cap=None,
        max_pe=None,
        min_revenue_growth=None,
        include_latam=None,
        include_technicals=False,
        min_eps_growth=None,
        min_buyback=None,
        min_score_threshold=None,
        max_results=None,
        sectors=None,
    )

    slow_note = _extract_stub_note(notes_slow)
    severity_slow, _, matched_slow = shared_notes.classify_note(slow_note)
    assert severity_slow == "info"
    assert matched_slow
    assert df_slow.attrs["_notes"][-1] == slow_note


def test_opportunities_flow_uses_preset_with_stub_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from application.screener import opportunities as opportunities_module
    from controllers import opportunities as controller_module
    from shared.errors import AppError

    _assert_distribution(opportunities_module._BASE_OPPORTUNITIES)

    preset_ready_dataset = [
        {
            "ticker": "NOVA",
            "payout_ratio": 10.0,
            "dividend_streak": 35,
            "cagr": 22.0,
            "dividend_yield": 1.6,
            "price": 210.0,
            "rsi": 50.0,
            "sma_50": 205.0,
            "sma_200": 198.0,
            "market_cap": 3_400.0,
            "pe_ratio": 24.0,
            "revenue_growth": 18.0,
            "is_latam": False,
            "trailing_eps": 4.8,
            "forward_eps": 5.4,
            "buyback": 9.0,
            "sector": "Technology",
        },
        {
            "ticker": "HEAL",
            "payout_ratio": 15.0,
            "dividend_streak": 32,
            "cagr": 20.0,
            "dividend_yield": 1.4,
            "price": 192.0,
            "rsi": 51.0,
            "sma_50": 188.0,
            "sma_200": 180.0,
            "market_cap": 3_100.0,
            "pe_ratio": 25.0,
            "revenue_growth": 16.0,
            "is_latam": False,
            "trailing_eps": 4.0,
            "forward_eps": 4.6,
            "buyback": 8.0,
            "sector": "Healthcare",
        },
        {
            "ticker": "LATM",
            "payout_ratio": 8.0,
            "dividend_streak": 30,
            "cagr": 21.0,
            "dividend_yield": 1.7,
            "price": 175.0,
            "rsi": 49.0,
            "sma_50": 172.0,
            "sma_200": 166.0,
            "market_cap": 2_800.0,
            "pe_ratio": 21.0,
            "revenue_growth": 19.0,
            "is_latam": True,
            "trailing_eps": 3.8,
            "forward_eps": 4.3,
            "buyback": 8.0,
            "sector": "Technology",
        },
        {
            "ticker": "VALUE",
            "payout_ratio": 55.0,
            "dividend_streak": 18,
            "cagr": 8.0,
            "dividend_yield": 2.6,
            "price": 120.0,
            "rsi": 57.0,
            "sma_50": 118.0,
            "sma_200": 112.0,
            "market_cap": 1_500.0,
            "pe_ratio": 27.0,
            "revenue_growth": 12.5,
            "is_latam": False,
            "trailing_eps": 3.5,
            "forward_eps": 3.7,
            "buyback": 0.2,
            "sector": "Industrials",
        },
    ]

    monkeypatch.setattr(
        opportunities_module, "_BASE_OPPORTUNITIES", preset_ready_dataset, raising=False
    )

    def _failing_yahoo(**_kwargs: object) -> None:
        raise AppError("timeout")

    monkeypatch.setattr(controller_module, "run_screener_yahoo", _failing_yahoo)

    app = _render_app()

    _set_selectbox(app, "Perfil recomendado", "Crecimiento balanceado")
    app.run()

    number_inputs = {element.label: element.value for element in app.get("number_input")}
    assert number_inputs["Capitalización mínima (US$ MM)"] == 1000
    assert number_inputs["P/E máximo"] == 28.0
    assert number_inputs["Crecimiento ingresos mínimo (%)"] == 12.0
    assert number_inputs["Buyback mínimo (%)"] == 1.0

    slider_values = {element.label: element.value for element in app.get("slider")}
    assert slider_values["Racha mínima de dividendos (años)"] == 5
    assert slider_values["Score mínimo"] == 72

    _click_search_button(app)

    app.run()

    dataframes = app.get("arrow_data_frame")
    assert dataframes, "Expected the results dataframe to be rendered"
    displayed = dataframes[0].value
    assert len(displayed) == 3
    assert set(displayed["ticker"]) == {"NOVA", "HEAL", "LATM"}
    assert displayed.iloc[0]["ticker"] == "NOVA"
    scores = pd.to_numeric(displayed["score_compuesto"], errors="coerce")
    assert not scores.isna().any()
    assert (scores >= 72).all()

    markdown_blocks = [element.value for element in app.get("markdown")]
    assert any("Datos simulados" in block for block in markdown_blocks)
    assert any("Filtros aplicados" in block for block in markdown_blocks)
    assert any("Stub procesó" in block for block in markdown_blocks)

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

    start = time.perf_counter()
    _click_search_button(app)
    app.run(timeout=10)
    elapsed = time.perf_counter() - start
    assert elapsed < 6.5, f"Execution took too long: {elapsed:.2f} seconds"

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
    assert any(
        "máximo solicitado" in block.lower() for block in markdown_blocks
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

    _click_search_button(app)

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


@pytest.mark.parametrize(
    ("max_results", "min_score_threshold"),
    [
        (2, 60),
        (3, 75),
        (1, 85),
    ],
)
def test_opportunities_flow_stub_failover_is_consistent_across_runs(
    monkeypatch: pytest.MonkeyPatch,
    max_results: int,
    min_score_threshold: int,
) -> None:
    from controllers import opportunities as controller_module

    base_rows = [
        {
            "ticker": "ALFA",
            "dividend_yield": 1.6,
            "payout_ratio": 18.0,
            "revenue_growth": 22.0,
            "buyback": 6.0,
            "market_cap": 4_500.0,
            "is_latam": False,
            "score_compuesto": 95.0,
        },
        {
            "ticker": "BETA",
            "dividend_yield": 1.4,
            "payout_ratio": 21.0,
            "revenue_growth": 20.0,
            "buyback": 5.5,
            "market_cap": 3_800.0,
            "is_latam": True,
            "score_compuesto": 91.0,
        },
        {
            "ticker": "GAMA",
            "dividend_yield": 1.8,
            "payout_ratio": 24.0,
            "revenue_growth": 19.5,
            "buyback": 5.0,
            "market_cap": 3_400.0,
            "is_latam": False,
            "score_compuesto": 88.0,
        },
        {
            "ticker": "DELTA",
            "dividend_yield": 1.2,
            "payout_ratio": 26.0,
            "revenue_growth": 17.0,
            "buyback": 4.8,
            "market_cap": 3_000.0,
            "is_latam": True,
            "score_compuesto": 83.0,
        },
        {
            "ticker": "OMEGA",
            "dividend_yield": 1.1,
            "payout_ratio": 28.0,
            "revenue_growth": 16.5,
            "buyback": 4.2,
            "market_cap": 2_900.0,
            "is_latam": False,
            "score_compuesto": 78.0,
        },
    ]

    base_frame = pd.DataFrame(base_rows).sort_values(
        "score_compuesto", ascending=False
    )
    call_history: list[Mapping[str, object]] = []

    def _stubbed_generate(filters: Optional[Mapping[str, object]] = None) -> Mapping[str, object]:
        captured = dict(filters or {})
        call_history.append(captured)

        requested_max = captured.get("max_results")
        requested_threshold = captured.get("min_score_threshold")

        max_count = int(requested_max) if requested_max is not None else None
        score_threshold = (
            float(requested_threshold) if requested_threshold is not None else 0.0
        )

        filtered = base_frame[base_frame["score_compuesto"] >= score_threshold]
        limited = filtered.head(max_count) if max_count else filtered
        table = limited.reset_index(drop=True).copy()

        notes = [
            (
                "Telemetría de contingencia: stub procesó "
                f"{len(filtered)} candidatos con umbral >= {score_threshold:.0f}."
            ),
            (
                "Se muestran "
                f"{len(table)} de {len(filtered)} resultados (máximo solicitado: {max_count or len(filtered)})."
            ),
            "Resultados generados por stub durante failover day.",
        ]

        return {"table": table, "notes": notes, "source": "stub"}

    monkeypatch.setattr(controller_module, "generate_opportunities_report", _stubbed_generate)

    app = _render_app()

    def _configure_growth_profile(app_test: AppTest) -> None:
        _set_number_input(app_test, "Capitalización mínima (US$ MM)", 2_500)
        _set_number_input(app_test, "P/E máximo", 30.0)
        _set_number_input(app_test, "Crecimiento ingresos mínimo (%)", 15.0)
        _set_checkbox(app_test, "Incluir Latam", False)
        _set_multiselect(app_test, "Sectores", ["Technology"])

    def _configure_balanced_preset(app_test: AppTest) -> None:
        _set_checkbox(app_test, "Incluir Latam", True)
        _set_multiselect(app_test, "Sectores", ["Technology", "Healthcare"])

    def _configure_value_rotation(app_test: AppTest) -> None:
        _set_number_input(app_test, "Capitalización mínima (US$ MM)", 1_000)
        _set_number_input(app_test, "P/E máximo", 26.0)
        _set_number_input(app_test, "Crecimiento ingresos mínimo (%)", 12.0)
        _set_checkbox(app_test, "Incluir Latam", True)
        _set_multiselect(app_test, "Sectores", ["Industrials", "Technology"])

    scenarios: list[Mapping[str, object]] = [
        {"configure": _configure_growth_profile},
        {"preset": "Crecimiento balanceado", "configure": _configure_balanced_preset},
        {"configure": _configure_value_rotation},
    ]

    previous_tickers: Optional[list[str]] = None

    for scenario in scenarios:
        preset_name = scenario.get("preset")
        if isinstance(preset_name, str):
            _set_selectbox(app, "Perfil recomendado", preset_name)
            app.run()

        configure: Callable[[AppTest], None] = scenario["configure"]
        configure(app)
        _set_number_input(app, "Máximo de resultados", max_results)
        _set_slider(app, "Score mínimo", min_score_threshold)

        app.run()

        table, captions, notes = _execute_search_cycle(app)

        assert len(table) <= max_results
        scores = pd.to_numeric(table["score_compuesto"], errors="coerce")
        assert not scores.isna().any()
        assert (scores >= min_score_threshold).all()

        assert any(
            "Resultados simulados" in caption for caption in captions
        ), "Expected stub fallback caption to remain visible"
        assert any(
            "Telemetría" in note or "failover" in note.lower() for note in notes
        ), "Expected telemetry note to be displayed"
        assert any(
            "máximo solicitado" in note for note in notes
        ), "Expected truncation note to mention the requested maximum"

        tickers = list(table["ticker"])
        if previous_tickers is None:
            previous_tickers = tickers
        else:
            assert tickers == previous_tickers, "Expected stable ordering across runs"

    assert len(call_history) == len(scenarios)
    for recorded_filters in call_history:
        recorded_max = recorded_filters.get("max_results")
        recorded_threshold = recorded_filters.get("min_score_threshold")
        assert recorded_max is not None, "Expected max_results to be forwarded to stub"
        assert recorded_threshold is not None, "Expected min_score_threshold to be forwarded"
        assert int(recorded_max) == max_results
        assert float(recorded_threshold) == pytest.approx(float(min_score_threshold))
