"""Contract tests for the portfolio Streamlit UI."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from controllers.portfolio.charts import render_basic_section
from controllers.portfolio.portfolio import render_portfolio_section
import controllers.portfolio.charts as charts_mod
from application.portfolio_service import PortfolioTotals
from domain.models import Controls
from shared.favorite_symbols import FavoriteSymbols
from services.portfolio_view import (
    PortfolioContributionMetrics,
    PortfolioViewSnapshot,
)


class _DummyContainer:
    def __enter__(self) -> "_DummyContainer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - standard context signature
        return None


class _ContextManager:
    def __init__(self, owner: "FakeStreamlit") -> None:
        self._owner = owner

    def __enter__(self) -> "_ContextManager":  # noqa: D401 - thin wrapper
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - thin wrapper
        return None

    def number_input(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner.number_input(*args, **kwargs)

    def selectbox(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner.selectbox(*args, **kwargs)

    def plotly_chart(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner.plotly_chart(*args, **kwargs)

    def info(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner.info(*args, **kwargs)

    def metric(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner.metric(*args, **kwargs)

    def line_chart(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner.line_chart(*args, **kwargs)

    def bar_chart(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner.bar_chart(*args, **kwargs)


class FakeStreamlit:
    """Minimal Streamlit stub capturing the interactions we care about."""

    def __init__(self, radio_sequence: Iterable[int], selectbox_defaults: dict[str, Any] | None = None) -> None:
        self.session_state: dict[str, Any] = {}
        self._radio_iter: Iterator[int] = iter(radio_sequence)
        self._selectbox_defaults = selectbox_defaults or {}
        self.radio_calls: list[dict[str, Any]] = []
        self.selectbox_calls: list[dict[str, Any]] = []
        self.number_input_calls: list[dict[str, Any]] = []
        self.subheaders: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.successes: list[str] = []
        self.plot_calls: list[dict[str, Any]] = []
        self.line_charts: list[pd.DataFrame] = []
        self.bar_charts: list[dict[str, Any]] = []
        self.metrics: list[tuple[Any, Any, Any]] = []
        self.markdowns: list[dict[str, Any]] = []

    # ---- Core widgets -------------------------------------------------
    def radio(
        self,
        label: str,
        *,
        options: Sequence[int],
        format_func,
        index: int = 0,
        horizontal: bool,
        **kwargs: Any,
    ) -> int:
        value = next(self._radio_iter)
        record = {"label": label, "options": list(options), "index": index}
        key = kwargs.get("key")
        if key is not None:
            record["key"] = key
            self.session_state[key] = value
        self.radio_calls.append(record)
        return value

    def selectbox(self, label: str, options: Sequence[Any], index: int = 0, key: str | None = None, **_: Any) -> Any:
        self.selectbox_calls.append({"label": label, "options": list(options), "key": key})
        if label in self._selectbox_defaults:
            return self._selectbox_defaults[label]
        return options[index] if options else None

    def number_input(self, label: str, *, min_value: Any, max_value: Any, value: Any, step: Any) -> Any:
        self.number_input_calls.append(
            {
                "label": label,
                "min_value": min_value,
                "max_value": max_value,
                "value": value,
                "step": step,
            }
        )
        return value

    def columns(self, layout: Sequence[Any] | int) -> list[_ContextManager]:
        if isinstance(layout, int):
            return [_ContextManager(self) for _ in range(layout)]
        return [_ContextManager(self) for _ in layout]

    def expander(self, label: str):  # noqa: ANN001 - mimics streamlit signature
        return _ContextManager(self)

    def spinner(self, *_: Any, **__: Any) -> _ContextManager:
        return _ContextManager(self)

    # ---- Feedback widgets ---------------------------------------------
    def subheader(self, text: str) -> None:
        self.subheaders.append(text)

    def info(self, message: str) -> None:
        self.warnings.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def success(self, message: str) -> None:
        self.successes.append(message)

    def caption(self, *_: Any, **__: Any) -> None:  # pragma: no cover - display only
        return None

    def plotly_chart(self, fig: Any, **kwargs: Any) -> None:
        self.plot_calls.append({"fig": fig, "kwargs": kwargs})

    def line_chart(self, data: pd.DataFrame) -> None:
        self.line_charts.append(data)

    def bar_chart(self, data: pd.DataFrame, **kwargs: Any) -> None:
        self.bar_charts.append({"data": data, "kwargs": kwargs})

    def write(self, *_: Any, **__: Any) -> None:
        return None

    def metric(self, label: str, value: Any, delta: Any | None = None) -> None:
        self.metrics.append((label, value, delta))

    def markdown(self, body: str, *, unsafe_allow_html: bool = False) -> None:
        self.markdowns.append({"body": body, "unsafe": unsafe_allow_html})

    def columns_context(self, layout: Sequence[Any]) -> None:  # pragma: no cover - helper for compatibility
        return None

    def select_slider(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused shim
        raise NotImplementedError

    # ---- Session helpers ----------------------------------------------
    def stop(self) -> None:  # pragma: no cover - not expected in these tests
        raise RuntimeError("streamlit.stop should not be called in tests")

    def rerun(self) -> None:  # pragma: no cover - not expected
        raise RuntimeError("streamlit.rerun should not be called in tests")


@pytest.fixture
def _portfolio_setup(monkeypatch: pytest.MonkeyPatch):
    import controllers.portfolio.portfolio as portfolio_mod

    def _configure(fake_st: FakeStreamlit, *, df_view: pd.DataFrame | None = None, all_symbols: list[str] | None = None):
        portfolio_mod.st = fake_st
        monkeypatch.setattr(portfolio_mod, "render_favorite_badges", lambda *a, **k: None)
        monkeypatch.setattr(portfolio_mod, "render_favorite_toggle", lambda *a, **k: None)

        class _FavoritesStub:
            def sort_options(self, options):
                return list(options)

            def default_index(self, options):
                return 0 if options else 0

            def format_symbol(self, sym):
                return sym

        monkeypatch.setattr(portfolio_mod, "get_persistent_favorites", lambda: _FavoritesStub())
        monkeypatch.setattr(portfolio_mod, "PortfolioService", lambda: MagicMock())
        monkeypatch.setattr(portfolio_mod, "TAService", lambda: MagicMock())

        df_positions = pd.DataFrame({"simbolo": ["GGAL"], "mercado": ["bcba"], "cantidad": [10], "costo_unitario": [100.0]})
        all_symbols = all_symbols or ["GGAL"]
        available_types = ["ACCION"]

        monkeypatch.setattr(
            portfolio_mod,
            "load_portfolio_data",
            lambda cli, svc: (df_positions.copy(), list(all_symbols), list(available_types)),
        )

        controls = Controls(
            refresh_secs=30,
            hide_cash=True,
            show_usd=False,
            order_by="valor_actual",
            desc=True,
            top_n=10,
            selected_syms=list(all_symbols),
            selected_types=list(available_types),
            symbol_query="",
        )
        monkeypatch.setattr(portfolio_mod, "render_sidebar", lambda *a, **k: controls)
        monkeypatch.setattr(portfolio_mod, "render_ui_controls", lambda: None)

        df_view = df_view or pd.DataFrame({"simbolo": ["GGAL"], "valor_actual": [1200.0]})

        def _fake_snapshot(*_args, **_kwargs):
            return PortfolioViewSnapshot(
                df_view=df_view,
                totals=PortfolioTotals(0.0, 0.0, 0.0, float("nan"), 0.0),
                apply_elapsed=0.0,
                totals_elapsed=0.0,
                generated_at=0.0,
                historical_total=pd.DataFrame(),
                contribution_metrics=PortfolioContributionMetrics.empty(),
            )

        monkeypatch.setattr(portfolio_mod.view_model_service, "get_portfolio_view", _fake_snapshot)

        basic = MagicMock()
        advanced = MagicMock()
        risk = MagicMock()
        fundamental = MagicMock()

        monkeypatch.setattr(portfolio_mod, "render_basic_section", basic)
        monkeypatch.setattr(portfolio_mod, "render_advanced_analysis", advanced)
        monkeypatch.setattr(portfolio_mod, "render_risk_analysis", risk)
        monkeypatch.setattr(portfolio_mod, "render_fundamental_analysis", fundamental)

        return portfolio_mod, basic, advanced, risk, fundamental

    return _configure


def test_render_portfolio_section_updates_tab_state(_portfolio_setup) -> None:
    fake_st = FakeStreamlit(radio_sequence=[1])
    portfolio_mod, basic, advanced, risk, fundamental = _portfolio_setup(fake_st)

    refresh_secs = render_portfolio_section(_DummyContainer(), cli=object(), fx_rates={})

    assert refresh_secs == 30
    assert fake_st.session_state["portfolio_tab"] == 1
    assert fake_st.radio_calls
    assert fake_st.radio_calls[0]["options"] == list(range(5))

    basic.assert_not_called()
    advanced.assert_called_once()
    risk.assert_not_called()
    fundamental.assert_not_called()


def test_render_portfolio_section_renders_symbol_selector_for_favorites(_portfolio_setup, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = FakeStreamlit(
        radio_sequence=[4],
        selectbox_defaults={"Seleccioná un símbolo (CEDEAR / ETF)": "GGAL", "Período": "6mo", "Intervalo": "1d"},
    )

    indicators_df = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0],
            "SMA_FAST": [100.0, 100.5, 101.0],
            "SMA_SLOW": [99.0, 99.5, 100.0],
            "BB_L": [95.0, 95.5, 96.0],
            "BB_U": [105.0, 105.5, 106.0],
            "MACD": [0.1, 0.2, 0.3],
            "MACD_SIGNAL": [0.05, 0.1, 0.15],
            "STOCH_K": [20.0, 40.0, 60.0],
            "STOCH_D": [15.0, 30.0, 45.0],
            "ICHI_CONV": [100.0, 100.5, 101.0],
            "ICHI_BASE": [99.0, 99.5, 100.0],
            "ATR": [1.0, 1.1, 1.2],
        }
    )

    class DummyTA:
        def fundamentals(self, ticker: str) -> dict:
            return {"ticker": ticker, "roe": 0.12}

        def indicators_for(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            return indicators_df.copy()

        def alerts_for(self, df_ind: pd.DataFrame) -> list[str]:
            return []

        def backtest(self, df_ind: pd.DataFrame, *, strategy: str = "sma") -> pd.DataFrame:
            return pd.DataFrame({"equity": [1.0, 1.05, 1.1]})

        def portfolio_history(self, *, simbolos: list[str], period: str = "1y") -> pd.DataFrame:
            return pd.DataFrame()

        def portfolio_fundamentals(self, simbolos: list[str]) -> pd.DataFrame:
            return pd.DataFrame()

    def _ta_factory():
        return DummyTA()

    portfolio_mod, basic, advanced, risk, fundamental = _portfolio_setup(fake_st, all_symbols=["GGAL", "AAPL"])
    monkeypatch.setattr(portfolio_mod, "TAService", _ta_factory)
    monkeypatch.setattr(portfolio_mod, "map_to_us_ticker", lambda sym: sym)
    monkeypatch.setattr(portfolio_mod, "render_fundamental_data", MagicMock())
    monkeypatch.setattr(portfolio_mod, "plot_technical_analysis_chart", lambda df, fast, slow: {"df": df, "fast": fast, "slow": slow})

    render_portfolio_section(_DummyContainer(), cli=object(), fx_rates={"ccl": 1000.0})

    assert fake_st.session_state["portfolio_tab"] == 4
    assert fake_st.selectbox_calls
    first_select = fake_st.selectbox_calls[0]
    assert first_select["label"] == "Seleccioná un símbolo (CEDEAR / ETF)"
    assert first_select["options"] == ["GGAL", "AAPL"]
    # Ensure advanced analysis helpers were not triggered in this branch
    advanced.assert_not_called()
    basic.assert_not_called()


def test_risk_analysis_ui_renders_new_charts(monkeypatch: pytest.MonkeyPatch) -> None:
    import controllers.portfolio.risk as risk_mod

    fake_st = FakeStreamlit(radio_sequence=[], selectbox_defaults={
        "Benchmark para beta y drawdown": "S&P 500 (^GSPC)",
        "Escenario": "Leve",
    })
    risk_mod.st = fake_st

    monkeypatch.setattr(risk_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "render_favorite_toggle", lambda *a, **k: None)

    df_view = pd.DataFrame({"simbolo": ["A", "B"], "valor_actual": [100.0, 200.0]})

    def fake_history(simbolos=None, period="1y"):
        if simbolos == ["S&P 500 (^GSPC)"] or simbolos == ["^GSPC"]:
            return pd.DataFrame({"^GSPC": [100.0, 102.0, 101.0, 103.0]})
        return pd.DataFrame({"A": [10.0, 10.5, 10.2, 10.8], "B": [20.0, 19.5, 20.5, 21.0]})

    tasvc = SimpleNamespace(portfolio_history=fake_history)

    fake_port_ret = pd.Series([0.01, -0.02, 0.03])
    asset_vols = pd.Series({"A": 0.12, "B": 0.09})
    asset_drawdowns = pd.Series({"A": -0.2, "B": -0.1})
    port_drawdown = -0.25

    monkeypatch.setattr(
        risk_mod,
        "compute_returns",
        lambda prices: prices.pct_change().dropna(),
    )
    monkeypatch.setattr(
        risk_mod,
        "compute_risk_metrics",
        lambda *a, **k: (
            0.2,
            1.1,
            0.05,
            pd.Series({"A": 0.6, "B": 0.4}),
            fake_port_ret,
            asset_vols,
            asset_drawdowns,
            port_drawdown,
        ),
    )

    drawdown_series_output = pd.Series([-0.01, -0.05], name="drawdown")
    monkeypatch.setattr(risk_mod, "drawdown_series", lambda *_: drawdown_series_output)

    class DummyFigure:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def update_layout(self, **_: Any) -> "DummyFigure":
            return self

        def update_traces(self, **_: Any) -> "DummyFigure":
            return self

        def update_xaxes(self, **_: Any) -> "DummyFigure":
            return self

        def update_yaxes(self, **_: Any) -> "DummyFigure":
            return self

        def add_vline(self, **_: Any) -> "DummyFigure":
            return self

    monkeypatch.setattr(risk_mod.px, "bar", lambda *a, **k: DummyFigure("volatility_dist"))

    def dummy_line(data, **kwargs):
        tag = "portfolio_drawdown" if data is drawdown_series_output else "rolling_vol"
        return DummyFigure(tag)

    monkeypatch.setattr(risk_mod.px, "line", dummy_line)

    class DummyScatter(DummyFigure):
        pass

    monkeypatch.setattr(risk_mod.px, "scatter", lambda *a, **k: DummyScatter("beta_scatter"))
    monkeypatch.setattr(
        risk_mod.px,
        "histogram",
        lambda *a, **k: DummyFigure("returns_hist"),
    )

    pm = risk_mod
    pm.render_risk_analysis(df_view, tasvc, favorites=FavoriteSymbols({}))

    tags = [call["fig"].tag for call in fake_st.plot_calls if hasattr(call["fig"], "tag")]
    assert {"volatility_dist", "portfolio_drawdown", "beta_scatter"}.issubset(tags)


def test_risk_analysis_ui_handles_missing_series(monkeypatch: pytest.MonkeyPatch) -> None:
    import controllers.portfolio.risk as risk_mod

    fake_st = FakeStreamlit(radio_sequence=[], selectbox_defaults={
        "Benchmark para beta y drawdown": "S&P 500 (^GSPC)",
        "Escenario": "Leve",
    })
    risk_mod.st = fake_st

    monkeypatch.setattr(risk_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "render_favorite_toggle", lambda *a, **k: None)

    df_view = pd.DataFrame({"simbolo": ["A"], "valor_actual": [100.0]})
    tasvc = SimpleNamespace(
        portfolio_history=lambda simbolos=None, period="1y": pd.DataFrame(
            {"A": [10.0, 10.0, 10.0], "^GSPC": [100.0, 101.0, 102.0]}
        )
    )

    monkeypatch.setattr(
        risk_mod,
        "compute_returns",
        lambda prices: prices.pct_change().dropna(),
    )
    monkeypatch.setattr(
        risk_mod,
        "compute_risk_metrics",
        lambda *a, **k: (
            0.0,
            float("nan"),
            0.0,
            pd.Series({"A": 1.0}),
            pd.Series([0.0, 0.0]),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            0.0,
        ),
    )
    monkeypatch.setattr(risk_mod, "drawdown_series", lambda *_: pd.Series(dtype=float))
    class DummyFigure:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def update_layout(self, **_: Any) -> "DummyFigure":
            return self

        def update_traces(self, **_: Any) -> "DummyFigure":
            return self

        def update_xaxes(self, **_: Any) -> "DummyFigure":
            return self

        def update_yaxes(self, **_: Any) -> "DummyFigure":
            return self

        def add_vline(self, **_: Any) -> "DummyFigure":
            return self

    monkeypatch.setattr(risk_mod.px, "bar", lambda *a, **k: DummyFigure("volatility_dist"))

    pm = risk_mod
    pm.render_risk_analysis(df_view, tasvc, favorites=FavoriteSymbols({}))

    assert any("volatilidad" in msg.lower() for msg in fake_st.warnings)
def test_render_advanced_analysis_controls_display(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = FakeStreamlit(radio_sequence=[])
    fake_st.checkbox = lambda *a, **k: False  # type: ignore[attr-defined]
    monkeypatch.setattr(charts_mod, "st", fake_st)
    monkeypatch.setattr(
        charts_mod,
        "compute_symbol_risk_metrics",
        lambda *a, **k: pd.DataFrame(),
    )
    monkeypatch.setattr(charts_mod, "plot_bubble_pl_vs_costo", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "plot_heat_pl_pct", lambda *a, **k: None)

    df = pd.DataFrame(
        {
            "simbolo": ["GGAL"],
            "tipo": ["Accion"],
            "valor_actual": [1000.0],
            "costo": [800.0],
            "pl": [200.0],
            "pl_%": [0.25],
            "pl_d": [5.0],
        }
    )

    charts_mod.render_advanced_analysis(df, tasvc=None)

    labels = [call["label"] for call in fake_st.selectbox_calls]
    assert "Período de métricas" in labels
    assert "Métrica de riesgo" in labels
    assert "Benchmark" in labels
def test_render_basic_section_renders_timeline_and_heatmap(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = FakeStreamlit(radio_sequence=[])

    import controllers.portfolio.charts as charts_mod

    class DummyFavorites:
        def sort_options(self, options):
            return list(options)

        def default_index(self, options):
            return 0 if options else 0

        def format_symbol(self, sym):
            return sym

    favorites_stub = DummyFavorites()

    monkeypatch.setattr(charts_mod, "st", fake_st)
    monkeypatch.setattr(charts_mod, "render_totals", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_table", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_favorite_toggle", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "get_persistent_favorites", lambda: favorites_stub)

    df_view = pd.DataFrame(
        {
            "simbolo": ["GGAL", "AL30"],
            "tipo": ["ACCION", "BONO"],
            "valor_actual": [1200.0, 800.0],
            "pl": [200.0, 50.0],
            "pl_d": [10.0, 5.0],
        }
    )
    controls = SimpleNamespace(order_by="valor_actual", desc=True, top_n=5, show_usd=False)
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "total_value": [1500.0, 1600.0, 1700.0],
            "total_cost": [1400.0, 1400.0, 1400.0],
        }
    )
    contributions = PortfolioContributionMetrics(
        by_symbol=pd.DataFrame(
            {
                "tipo": ["ACCION", "BONO"],
                "simbolo": ["GGAL", "AL30"],
                "valor_actual": [1200.0, 800.0],
                "costo": [1000.0, 750.0],
                "pl": [200.0, 50.0],
                "pl_d": [10.0, 5.0],
                "valor_actual_pct": [60.0, 40.0],
                "pl_pct": [80.0, 20.0],
            }
        ),
        by_type=pd.DataFrame(
            {
                "tipo": ["ACCION", "BONO"],
                "valor_actual": [1200.0, 800.0],
                "costo": [1000.0, 750.0],
                "pl": [200.0, 50.0],
                "pl_d": [10.0, 5.0],
                "valor_actual_pct": [60.0, 40.0],
                "pl_pct": [80.0, 20.0],
            }
        ),
    )

    render_basic_section(
        df_view,
        controls,
        ccl_rate=1000.0,
        totals=None,
        favorites=None,
        historical_total=history,
        contribution_metrics=contributions,
    )

    plotted_keys = {call["kwargs"].get("key") for call in fake_st.plot_calls}
    assert {"portfolio_timeline", "portfolio_contribution_heatmap", "portfolio_contribution_table"}.issubset(
        plotted_keys
    )
    assert not any("históricos" in msg for msg in fake_st.warnings)


def test_render_basic_section_handles_missing_analytics(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = FakeStreamlit(radio_sequence=[])

    import controllers.portfolio.charts as charts_mod

    class DummyFavorites:
        def sort_options(self, options):
            return list(options)

        def default_index(self, options):
            return 0 if options else 0

        def format_symbol(self, sym):
            return sym

    favorites_stub = DummyFavorites()

    monkeypatch.setattr(charts_mod, "st", fake_st)
    monkeypatch.setattr(charts_mod, "render_totals", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_table", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_favorite_toggle", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "get_persistent_favorites", lambda: favorites_stub)

    df_view = pd.DataFrame(
        {
            "simbolo": ["GGAL"],
            "tipo": ["ACCION"],
            "valor_actual": [1200.0],
            "pl": [200.0],
            "pl_d": [10.0],
        }
    )
    controls = SimpleNamespace(order_by="valor_actual", desc=True, top_n=5, show_usd=False)

    render_basic_section(
        df_view,
        controls,
        ccl_rate=1000.0,
        totals=None,
        favorites=None,
        historical_total=pd.DataFrame(),
        contribution_metrics=PortfolioContributionMetrics.empty(),
    )

    info_messages = " ".join(fake_st.warnings)
    assert "históricos" in info_messages
    assert "contribución" in info_messages
