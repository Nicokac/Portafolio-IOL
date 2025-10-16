"""Contract tests for the portfolio Streamlit UI."""
from __future__ import annotations

import base64
import zipfile
import sys
import xml.etree.ElementTree as ET
from io import BytesIO
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
from services.notifications import NotificationFlags
from shared.favorite_symbols import FavoriteSymbols
from ui.notifications import tab_badge_label, tab_badge_suffix
from services.portfolio_view import (
    PortfolioContributionMetrics,
    PortfolioViewSnapshot,
)
from shared.portfolio_export import PortfolioSnapshotExport


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


class _Placeholder:
    def __init__(self, owner: "FakeStreamlit") -> None:
        self._owner = owner

    def empty(self) -> None:
        return None

    def container(self) -> _ContextManager:
        return _ContextManager(self._owner)

    def markdown(self, body: str, *, unsafe_allow_html: bool = False) -> None:
        self._owner.markdowns.append({
            "body": body,
            "unsafe": unsafe_allow_html,
            "placeholder": True,
        })

    def info(self, message: str) -> None:
        self._owner.info(message)

    def caption(self, text: str) -> None:
        self._owner.caption(text)


class FakeStreamlit:
    """Minimal Streamlit stub capturing the interactions we care about."""

    def __init__(
        self,
        radio_sequence: Iterable[int],
        selectbox_defaults: dict[str, Any] | None = None,
        *,
        multiselect_responses: dict[str, Sequence[Any]] | None = None,
        checkbox_values: dict[str, bool] | None = None,
        slider_values: dict[str, Any] | None = None,
    ) -> None:
        self.session_state: dict[str, Any] = {}
        self._radio_iter: Iterator[int] = iter(radio_sequence)
        self._selectbox_defaults = selectbox_defaults or {}
        self._multiselect_responses = {
            key: list(value) for key, value in (multiselect_responses or {}).items()
        }
        self._checkbox_values = dict(checkbox_values or {})
        self._slider_values = dict(slider_values or {})
        self.radio_calls: list[dict[str, Any]] = []
        self.selectbox_calls: list[dict[str, Any]] = []
        self.multiselect_calls: list[dict[str, Any]] = []
        self.number_input_calls: list[dict[str, Any]] = []
        self.subheaders: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.successes: list[str] = []
        self.plot_calls: list[dict[str, Any]] = []
        self.line_charts: list[pd.DataFrame] = []
        self.bar_charts: list[dict[str, Any]] = []
        self.metrics: list[tuple[Any, Any, Any, dict[str, Any]]] = []
        self.markdowns: list[dict[str, Any]] = []
        self.captions: list[str] = []
        self.checkbox_calls: list[dict[str, Any]] = []
        self.slider_calls: list[dict[str, Any]] = []
        self.download_buttons: list[dict[str, Any]] = []
        self._placeholders: list[_Placeholder] = []

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
        display_labels = [format_func(opt) for opt in options]
        record = {
            "label": label,
            "options": list(options),
            "index": index,
            "display_labels": display_labels,
        }
        key = kwargs.get("key")
        if key is not None:
            record["key"] = key
            self.session_state[key] = value
        self.radio_calls.append(record)
        return value

    def selectbox(self, label: str, options: Sequence[Any], index: int = 0, key: str | None = None, **_: Any) -> Any:
        self.selectbox_calls.append({"label": label, "options": list(options), "key": key})
        if label in self._selectbox_defaults:
            result = self._selectbox_defaults[label]
        else:
            result = options[index] if options else None
        if key is not None:
            self.session_state[key] = result
        return result

    def multiselect(
        self,
        label: str,
        options: Sequence[Any],
        *,
        default: Sequence[Any] | None = None,
        format_func=lambda x: x,
        key: str | None = None,
    ) -> list[Any]:
        rendered = [format_func(opt) for opt in options] if format_func else list(options)
        record = {
            "label": label,
            "options": list(options),
            "rendered": rendered,
            "default": list(default) if default is not None else [],
            "key": key,
        }
        self.multiselect_calls.append(record)
        state_key = key or label
        if key and key in self.session_state:
            selection = self.session_state[key]
        elif state_key in self._multiselect_responses:
            selection = list(self._multiselect_responses[state_key])
        else:
            selection = list(default) if default is not None else []
        if key is not None:
            self.session_state[key] = list(selection)
        return list(selection)

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

    def checkbox(self, label: str, *, value: bool = False, key: str | None = None) -> bool:
        record = {"label": label, "value": value, "key": key}
        self.checkbox_calls.append(record)
        state_key = key or label
        result = self._checkbox_values.get(state_key, value)
        if key is not None:
            self.session_state[key] = result
        return result

    def slider(
        self,
        label: str,
        *,
        min_value: Any,
        max_value: Any,
        value: Any,
        step: Any,
        key: str | None = None,
    ) -> Any:
        record = {
            "label": label,
            "min_value": min_value,
            "max_value": max_value,
            "value": value,
            "step": step,
            "key": key,
        }
        self.slider_calls.append(record)
        state_key = key or label
        result = self._slider_values.get(state_key, value)
        if key is not None:
            self.session_state[key] = result
        return result

    def columns(self, layout: Sequence[Any] | int) -> list[_ContextManager]:
        if isinstance(layout, int):
            return [_ContextManager(self) for _ in range(layout)]
        return [_ContextManager(self) for _ in layout]

    def container(self) -> _ContextManager:
        return _ContextManager(self)

    def tabs(self, labels: Sequence[str]) -> list[_ContextManager]:
        return [_ContextManager(self) for _ in labels]

    def expander(self, label: str, *_, **__):  # noqa: ANN001 - mimics streamlit signature
        return _ContextManager(self)

    def spinner(self, *_: Any, **__: Any) -> _ContextManager:
        return _ContextManager(self)

    def empty(self) -> _Placeholder:
        placeholder = _Placeholder(self)
        self._placeholders.append(placeholder)
        return placeholder

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

    def caption(self, text: str, *_: Any, **__: Any) -> None:
        self.captions.append(text)

    def plotly_chart(self, fig: Any, **kwargs: Any) -> None:
        self.plot_calls.append({"fig": fig, "kwargs": kwargs})

    def line_chart(self, data: pd.DataFrame) -> None:
        self.line_charts.append(data)

    def bar_chart(self, data: pd.DataFrame, **kwargs: Any) -> None:
        self.bar_charts.append({"data": data, "kwargs": kwargs})

    def write(self, *_: Any, **__: Any) -> None:
        return None

    def metric(
        self,
        label: str,
        value: Any,
        delta: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.metrics.append((label, value, delta, kwargs))

    def markdown(self, body: str, *, unsafe_allow_html: bool = False) -> None:
        self.markdowns.append({"body": body, "unsafe": unsafe_allow_html})

    def download_button(
        self,
        label: str,
        data: Any,
        *,
        file_name: str,
        mime: str,
        key: str | None = None,
    ) -> None:
        self.download_buttons.append(
            {
                "label": label,
                "data": data,
                "file_name": file_name,
                "mime": mime,
                "key": key,
            }
        )
        if key is not None:
            self.session_state[key] = data

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

    def _configure(
        fake_st: FakeStreamlit,
        *,
        df_view: pd.DataFrame | None = None,
        all_symbols: list[str] | None = None,
        notifications: NotificationFlags | None = None,
    ):
        portfolio_mod.st = fake_st
        portfolio_mod.reset_portfolio_services()
        portfolio_mod._INCREMENTAL_CACHE.clear()
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

        def _view_model_factory():
            return SimpleNamespace(get_portfolio_view=_fake_snapshot)

        basic = MagicMock()
        advanced = MagicMock()
        risk = MagicMock()
        fundamental = MagicMock()
        technical_badge = MagicMock()

        def _summary_stub(*args, **kwargs):
            basic()
            return True

        def _table_stub(*args, **kwargs):
            basic()

        def _charts_stub(*args, **kwargs):
            basic()

        monkeypatch.setattr(portfolio_mod, "render_summary_section", _summary_stub)
        monkeypatch.setattr(portfolio_mod, "render_table_section", _table_stub)
        monkeypatch.setattr(portfolio_mod, "render_charts_section", _charts_stub)
        monkeypatch.setattr(portfolio_mod, "render_advanced_analysis", advanced)
        monkeypatch.setattr(portfolio_mod, "render_risk_analysis", risk)
        monkeypatch.setattr(portfolio_mod, "render_fundamental_analysis", fundamental)
        monkeypatch.setattr(portfolio_mod, "render_technical_badge", technical_badge)
        def _notifications_factory():
            return SimpleNamespace(get_flags=lambda: notifications or NotificationFlags())

        return (
            portfolio_mod,
            basic,
            advanced,
            risk,
            fundamental,
            technical_badge,
            _view_model_factory,
            _notifications_factory,
        )

    return _configure


@pytest.fixture
def sample_export_snapshot() -> dict[str, Any]:
    """Provide a representative snapshot payload for export tests."""

    from shared.test.test_portfolio_export import _snapshot

    base = _snapshot()

    totals_raw = base.totals or {}
    totals = PortfolioTotals(
        totals_raw.get("total_value", 0.0) or 0.0,
        totals_raw.get("total_cost", 0.0) or 0.0,
        totals_raw.get("total_pl", 0.0) or 0.0,
        totals_raw.get("total_pl_pct", 0.0) or 0.0,
        totals_raw.get("total_cash", 0.0) or 0.0,
    )

    df_view = base.positions.copy()
    history = base.history.copy()
    contributions = PortfolioContributionMetrics(
        by_symbol=base.contributions_by_symbol.copy(),
        by_type=base.contributions_by_type.copy(),
    )

    class SnapshotDouble:
        def __init__(self) -> None:
            self.df_view = df_view
            self.totals = totals
            self.historical_total = history
            self.contribution_metrics = contributions
            self.generated_at = 0.0

    return {
        "snapshot": SnapshotDouble(),
        "df_view": df_view,
        "totals": totals,
        "historical_total": history,
        "contribution_metrics": contributions,
    }


def test_render_portfolio_section_updates_tab_state(_portfolio_setup) -> None:
    fake_st = FakeStreamlit(radio_sequence=[1])
    (
        portfolio_mod,
        basic,
        advanced,
        risk,
        fundamental,
        technical_badge,
        view_model_factory,
        notifications_factory,
    ) = _portfolio_setup(fake_st)

    refresh_secs = render_portfolio_section(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert refresh_secs == 30
    assert fake_st.session_state["portfolio_tab"] == 1
    assert fake_st.radio_calls
    assert fake_st.radio_calls[0]["options"] == list(range(5))

    basic.assert_not_called()
    advanced.assert_called_once()
    risk.assert_not_called()
    fundamental.assert_not_called()


def test_render_portfolio_section_tab_labels_without_flags(_portfolio_setup) -> None:
    fake_st = FakeStreamlit(radio_sequence=[0])
    (
        portfolio_mod,
        *_rest,
        view_model_factory,
        notifications_factory,
    ) = _portfolio_setup(fake_st)

    render_portfolio_section(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    labels = fake_st.radio_calls[0]["display_labels"]
    suffixes = {
        tab_badge_suffix("risk").strip(),
        tab_badge_suffix("earnings").strip(),
        tab_badge_suffix("technical").strip(),
        tab_badge_label("risk"),
        tab_badge_label("earnings"),
        tab_badge_label("technical"),
    }
    for label in labels:
        assert not any(token and token in label for token in suffixes)


def test_render_portfolio_section_applies_tab_badges_when_flags_active(_portfolio_setup) -> None:
    fake_st = FakeStreamlit(radio_sequence=[2])
    flags = NotificationFlags(risk_alert=True, technical_signal=True, upcoming_earnings=True)
    (
        portfolio_mod,
        basic,
        advanced,
        risk,
        fundamental,
        technical_badge,
        view_model_factory,
        notifications_factory,
    ) = _portfolio_setup(fake_st, notifications=flags)

    render_portfolio_section(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    labels = fake_st.radio_calls[0]["display_labels"]
    assert labels[2].endswith(f"{tab_badge_suffix('risk')} {tab_badge_label('risk')}")
    assert labels[3].endswith(f"{tab_badge_suffix('earnings')} {tab_badge_label('earnings')}")
    assert labels[4].endswith(f"{tab_badge_suffix('technical')} {tab_badge_label('technical')}")

    risk.assert_called_once()
    assert risk.call_args.kwargs.get("notifications") == flags
    technical_badge.assert_not_called()


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

    (
        portfolio_mod,
        basic,
        advanced,
        risk,
        fundamental,
        technical_badge,
        view_model_factory,
        notifications_factory,
    ) = _portfolio_setup(
        fake_st,
        all_symbols=["GGAL", "AAPL"],
        notifications=NotificationFlags(technical_signal=True),
    )
    monkeypatch.setattr(portfolio_mod, "TAService", _ta_factory)
    monkeypatch.setattr(portfolio_mod, "map_to_us_ticker", lambda sym: sym)
    monkeypatch.setattr(portfolio_mod, "render_fundamental_data", MagicMock())
    monkeypatch.setattr(portfolio_mod, "plot_technical_analysis_chart", lambda df, fast, slow: {"df": df, "fast": fast, "slow": slow})

    render_portfolio_section(
        _DummyContainer(),
        cli=object(),
        fx_rates={"ccl": 1000.0},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert fake_st.session_state["portfolio_tab"] == 4
    assert fake_st.selectbox_calls
    symbol_select = next(
        call for call in fake_st.selectbox_calls if call["label"] == "Seleccioná un símbolo (CEDEAR / ETF)"
    )
    assert symbol_select["options"] == ["GGAL", "AAPL"]
    # Ensure advanced analysis helpers were not triggered in this branch
    advanced.assert_not_called()
    basic.assert_not_called()
    technical_badge.assert_called_once()


def test_risk_analysis_ui_renders_new_charts(monkeypatch: pytest.MonkeyPatch) -> None:
    import controllers.portfolio.risk as risk_mod

    fake_st = FakeStreamlit(
        radio_sequence=[],
        selectbox_defaults={
            "Benchmark para beta y drawdown": "S&P 500 (^GSPC)",
            "Ventana para correlaciones móviles": "3 meses (63)",
            "Nivel de confianza para VaR/CVaR": "95%",
            "Tipo de activo a incluir": "ACCION",
        },
    )
    risk_mod.st = fake_st
    fake_st.session_state["selected_asset_types"] = ["ACCION"]

    monkeypatch.setattr(risk_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "render_favorite_toggle", lambda *a, **k: None)

    class _CacheStub:
        def get_history(self, *_args, loader=None, **_kwargs):
            if callable(loader):
                return loader()
            return pd.DataFrame()

    monkeypatch.setattr(risk_mod, "get_market_data_cache", lambda: _CacheStub())

    df_view = pd.DataFrame(
        {
            "simbolo": ["A1", "A2", "B"],
            "valor_actual": [100.0, 150.0, 200.0],
            "tipo": ["ACCION", "ACCION", "BONO"],
        }
    )

    history_calls: list[dict[str, Any]] = []

    price_series = {
        "A1": [10.0, 10.5, 10.2, 10.8],
        "A2": [20.0, 19.5, 20.5, 21.0],
    }

    def fake_history(simbolos=None, period="1y"):
        symbols = list(simbolos or [])
        call = {"simbolos": symbols, "period": period}
        history_calls.append(call)
        if symbols == ["S&P 500 (^GSPC)"] or symbols == ["^GSPC"]:
            return pd.DataFrame({"^GSPC": [100.0, 102.0, 101.0, 103.0]})
        relevant = [sym for sym in symbols if sym in price_series]
        if not relevant:
            return pd.DataFrame()
        return pd.DataFrame({sym: price_series[sym] for sym in relevant})

    tasvc = SimpleNamespace(portfolio_history=fake_history)

    fake_port_ret = pd.Series([0.01, -0.02, 0.03])
    asset_vols = pd.Series({"A1": 0.12, "A2": 0.09})
    asset_drawdowns = pd.Series({"A1": -0.2, "A2": -0.1})
    port_drawdown = -0.25

    monkeypatch.setattr(
        risk_mod,
        "compute_returns",
        lambda prices: prices.pct_change().dropna(),
    )
    monkeypatch.setattr(risk_mod, "beta", lambda returns, bench: 0.75 if not returns.empty else float("nan"))
    def fake_compute(*args, **kwargs):
        assert kwargs.get("var_confidence") == 0.95
        return (
            0.2,
            1.1,
            0.05,
            0.07,
                pd.Series({"A1": 0.6, "A2": 0.4}),
            fake_port_ret,
            asset_vols,
            asset_drawdowns,
            port_drawdown,
        )

    monkeypatch.setattr(risk_mod, "compute_risk_metrics", fake_compute)
    monkeypatch.setattr(
        risk_mod,
        "monte_carlo_simulation",
        lambda *a, **k: pd.DataFrame({"sim": [1.0, 1.02, 1.05]}),
    )
    monkeypatch.setattr(
        risk_mod,
        "benchmark_analysis",
        lambda *a, **k: {
            "tracking_error": 0.02,
            "active_return": 0.01,
            "information_ratio": 0.5,
            "factor_betas": {"MKT": 1.1},
            "r_squared": 0.8,
        },
    )

    drawdown_series_output = pd.Series([-0.01, -0.05], name="drawdown")
    monkeypatch.setattr(risk_mod, "drawdown_series", lambda *_: drawdown_series_output)

    rolling_corr_output = pd.DataFrame(
        {
            "A1↔A2": [0.1, 0.2, 0.3],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    monkeypatch.setattr(risk_mod, "rolling_correlations", lambda *_: rolling_corr_output)

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
        if data is drawdown_series_output:
            tag = "portfolio_drawdown"
        elif data is rolling_corr_output:
            tag = "rolling_corr"
        else:
            tag = "rolling_vol"
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
    expected_tags = {"volatility_dist", "portfolio_drawdown", "rolling_corr", "returns_hist"}
    assert expected_tags.issubset(tags)
    assert any("CVaR" in label for label, *_ in fake_st.metrics)
    labels = [call["label"] for call in fake_st.selectbox_calls]
    assert "Calcular correlación sobre el último período:" in labels
    filtered_calls = [
        call for call in history_calls if call["simbolos"] and not call["simbolos"][0].startswith("^")
    ]
    assert filtered_calls, "Expected history calls for portfolio symbols"
    assert all(
        set(call["simbolos"]).issubset({"A1", "A2"}) for call in filtered_calls
    ), filtered_calls
    assert all("B" not in call["simbolos"] for call in filtered_calls)


def test_risk_analysis_ui_handles_missing_series(monkeypatch: pytest.MonkeyPatch) -> None:
    import controllers.portfolio.risk as risk_mod

    fake_st = FakeStreamlit(
        radio_sequence=[],
        selectbox_defaults={
            "Benchmark para beta y drawdown": "S&P 500 (^GSPC)",
            "Nivel de confianza para VaR/CVaR": "95%",
            "Tipo de activo a incluir": "Todos",
        },
    )
    risk_mod.st = fake_st
    fake_st.session_state["selected_asset_types"] = ["ACCION", "BONO"]

    monkeypatch.setattr(risk_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "render_favorite_toggle", lambda *a, **k: None)

    df_view = pd.DataFrame(
        {"simbolo": ["A"], "valor_actual": [100.0], "tipo": ["ACCION"]}
    )
    def single_history(simbolos=None, period="1y"):
        if simbolos and simbolos[0] == "^GSPC":
            return pd.DataFrame({"^GSPC": [100.0, 101.0, 102.0]})
        return pd.DataFrame({"A": [10.0, 10.0, 10.0]})

    tasvc = SimpleNamespace(portfolio_history=single_history)

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
    pm.render_risk_analysis(
        df_view,
        tasvc,
        favorites=FavoriteSymbols({}),
        available_types=["ACCION", "BONO"],
    )

    assert any("volatilidad" in msg.lower() for msg in fake_st.warnings)


def test_risk_analysis_warns_when_selected_type_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import controllers.portfolio.risk as risk_mod

    fake_st = FakeStreamlit(
        radio_sequence=[],
        selectbox_defaults={"Tipo de activo a incluir": "BONO"},
    )
    risk_mod.st = fake_st
    fake_st.session_state["selected_asset_types"] = ["BONO"]

    monkeypatch.setattr(risk_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "render_favorite_toggle", lambda *a, **k: None)

    df_view = pd.DataFrame(
        {"simbolo": ["A"], "valor_actual": [100.0], "tipo": ["ACCION"]}
    )

    def _should_not_run(*_, **__):
        raise AssertionError("No debería consultarse histórico cuando no hay datos para el tipo")

    tasvc = SimpleNamespace(portfolio_history=_should_not_run)

    risk_mod.render_risk_analysis(
        df_view,
        tasvc,
        favorites=FavoriteSymbols({}),
        available_types=["ACCION", "BONO"],
    )

    assert any("No hay datos para los tipos seleccionados" in msg for msg in fake_st.warnings)
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
    monkeypatch.setattr(charts_mod, "render_portfolio_exports", lambda *a, **k: None)
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
    monkeypatch.setattr(charts_mod, "render_portfolio_exports", lambda *a, **k: None)

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


def test_render_portfolio_exports_offers_zip_and_excel(
    monkeypatch: pytest.MonkeyPatch, sample_export_snapshot: dict[str, Any]
) -> None:
    fake_st = FakeStreamlit(
        radio_sequence=[],
        multiselect_responses={
            "metrics_demo": ["total_value", "cash_ratio"],
            "charts_demo": ["pl_top"],
        },
        checkbox_values={"rankings_demo": True, "history_demo": False},
        slider_values={"limit_demo": 15},
    )

    import ui.export as export_mod

    monkeypatch.setattr(export_mod, "st", fake_st)

    csv_call: dict[str, Any] = {}
    excel_call: dict[str, Any] = {}

    def fake_create_csv_bundle(snapshot: PortfolioSnapshotExport, **kwargs: Any) -> bytes:
        csv_call.update({"snapshot": snapshot, **kwargs})
        return b"csv-bytes"

    def fake_create_excel_workbook(snapshot: PortfolioSnapshotExport, **kwargs: Any) -> bytes:
        excel_call.update({"snapshot": snapshot, **kwargs})
        return b"excel-bytes"

    monkeypatch.setattr(export_mod, "create_csv_bundle", fake_create_csv_bundle)
    monkeypatch.setattr(export_mod, "create_excel_workbook", fake_create_excel_workbook)

    export_mod.render_portfolio_exports(
        snapshot=sample_export_snapshot["snapshot"],
        df_view=sample_export_snapshot["df_view"],
        totals=sample_export_snapshot["totals"],
        historical_total=sample_export_snapshot["historical_total"],
        contribution_metrics=sample_export_snapshot["contribution_metrics"],
        filename_prefix="demo",
    )

    assert csv_call
    assert excel_call
    assert isinstance(csv_call["snapshot"], PortfolioSnapshotExport)
    assert csv_call["metric_keys"] == ["total_value", "cash_ratio"]
    assert csv_call["include_rankings"] is True
    assert csv_call["include_history"] is False
    assert csv_call["limit"] == 15

    assert excel_call["metric_keys"] == ["total_value", "cash_ratio"]
    assert excel_call["chart_keys"] == ["pl_top"]
    assert excel_call["include_rankings"] is True
    assert excel_call["include_history"] is False
    assert excel_call["limit"] == 15

    assert len(fake_st.download_buttons) == 2

    csv_button = next(btn for btn in fake_st.download_buttons if btn["mime"] == "application/zip")
    assert csv_button["label"] == "⬇️ Descargar CSV (ZIP)"
    assert csv_button["data"] == b"csv-bytes"
    assert csv_button["file_name"].endswith("_analisis.zip")

    excel_button = next(
        btn
        for btn in fake_st.download_buttons
        if btn["mime"]
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert excel_button["label"] == "⬇️ Descargar Excel (.xlsx)"
    assert excel_button["data"] == b"excel-bytes"
    assert excel_button["file_name"].endswith("_analisis.xlsx")


def test_render_portfolio_exports_warns_without_kaleido(
    monkeypatch: pytest.MonkeyPatch, sample_export_snapshot: dict[str, Any]
) -> None:
    fake_st = FakeStreamlit(radio_sequence=[])

    import ui.export as export_mod

    monkeypatch.setattr(export_mod, "st", fake_st)
    monkeypatch.setattr(
        export_mod,
        "create_csv_bundle",
        lambda snapshot, **kwargs: b"csv-bytes",
    )

    def boom(*_args: Any, **_kwargs: Any) -> bytes:
        raise ValueError("kaleido missing")

    monkeypatch.setattr(export_mod, "create_excel_workbook", boom)

    export_mod.render_portfolio_exports(
        snapshot=sample_export_snapshot["snapshot"],
        df_view=sample_export_snapshot["df_view"],
        totals=sample_export_snapshot["totals"],
        historical_total=sample_export_snapshot["historical_total"],
        contribution_metrics=sample_export_snapshot["contribution_metrics"],
        filename_prefix="demo",
    )

    assert len(fake_st.download_buttons) == 1
    assert fake_st.download_buttons[0]["mime"] == "application/zip"
    assert any("kaleido" in message.lower() for message in fake_st.warnings)


def test_render_portfolio_exports_generates_full_package(
    monkeypatch: pytest.MonkeyPatch,
    sample_export_snapshot: dict[str, Any],
) -> None:
    import ui.export as export_mod

    chart_keys = [spec.key for spec in export_mod.CHART_SPECS[:3]]
    if len(chart_keys) < 2:
        pytest.skip("Se requieren múltiples gráficos configurados para validar la exportación completa")

    fake_st = FakeStreamlit(
        radio_sequence=[],
        multiselect_responses={
            "metrics_portafolio": ["total_value", "total_pl", "cash_ratio"],
            "charts_portafolio": chart_keys,
        },
        checkbox_values={"rankings_portafolio": True, "history_portafolio": True},
        slider_values={"limit_portafolio": 25},
    )

    monkeypatch.setattr(export_mod, "st", fake_st)

    png_bytes = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    )

    def fake_to_image(fig, *, format="png", **kwargs):
        assert format == "png"
        assert "scope" not in kwargs
        return png_bytes

    monkeypatch.setattr("shared.export._get_kaleido_scope", lambda: object())
    monkeypatch.setattr("shared.export.pio.to_image", fake_to_image)

    export_mod.render_portfolio_exports(
        snapshot=sample_export_snapshot["snapshot"],
        df_view=sample_export_snapshot["df_view"],
        totals=sample_export_snapshot["totals"],
        historical_total=sample_export_snapshot["historical_total"],
        contribution_metrics=sample_export_snapshot["contribution_metrics"],
        filename_prefix="portafolio",
    )

    assert len(fake_st.download_buttons) == 2

    csv_button = next(btn for btn in fake_st.download_buttons if btn["mime"] == "application/zip")
    with zipfile.ZipFile(BytesIO(csv_button["data"])) as zf:
        members = set(zf.namelist())
    assert {"kpis.csv", "positions.csv"}.issubset(members)
    assert any(name.startswith("ranking_") for name in members)

    excel_button = next(
        btn
        for btn in fake_st.download_buttons
        if btn["mime"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    with zipfile.ZipFile(BytesIO(excel_button["data"])) as zf:
        workbook_files = set(zf.namelist())
        assert "xl/workbook.xml" in workbook_files
        drawing_files = [name for name in workbook_files if name.startswith("xl/drawings/drawing")]
        assert drawing_files
        drawing_xml = ET.fromstring(zf.read(drawing_files[0]))
    ns = {"xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"}
    anchors = drawing_xml.findall("xdr:twoCellAnchor", ns) + drawing_xml.findall("xdr:oneCellAnchor", ns)
    assert len(anchors) >= 2

    metric_call = next(call for call in fake_st.multiselect_calls if call["label"].startswith("Métricas"))
    chart_call = next(call for call in fake_st.multiselect_calls if call["label"].startswith("Gráficos"))
    assert set(metric_call["options"]) >= {"total_value", "total_pl", "cash_ratio"}
    assert set(chart_call["options"]) >= set(chart_keys)
    assert fake_st.session_state["metrics_portafolio"] == ["total_value", "total_pl", "cash_ratio"]
    assert fake_st.session_state["charts_portafolio"] == list(chart_keys)
