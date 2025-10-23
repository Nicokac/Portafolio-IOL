"""Contract tests for the portfolio Streamlit UI."""

from types import SimpleNamespace
from typing import Any, Iterable
from unittest.mock import MagicMock

import pandas as pd
import pytest

import controllers.portfolio.charts as charts_mod
import ui.export as export_mod
import ui.tables as tables_mod
from application.portfolio_service import PortfolioTotals
from controllers.portfolio.charts import render_basic_section
from controllers.portfolio.portfolio import render_portfolio_section
from domain.models import Controls
from services.notifications import NotificationFlags
from services.portfolio_view import (
    PortfolioContributionMetrics,
    PortfolioViewSnapshot,
)
from shared.favorite_symbols import FavoriteSymbols
from tests.fixtures.common import DummyCtx
from tests.fixtures.streamlit import UIFakeStreamlit as FakeStreamlit
from tests.fixtures.streamlit import _ContextManager


IOL_ASSET_TYPES: list[str] = ["Cedear", "Acciones", "Bono", "Letra", "FCI"]
from ui.notifications import tab_badge_label, tab_badge_suffix


@pytest.fixture
def _portfolio_setup(monkeypatch: pytest.MonkeyPatch):
    import controllers.portfolio.portfolio as portfolio_mod

    def _configure(
        fake_st: FakeStreamlit,
        *,
        df_view: pd.DataFrame | None = None,
        all_symbols: list[str] | None = None,
        notifications: NotificationFlags | None = None,
        available_types: Iterable[str] | None = None,
        totals: PortfolioTotals | None = None,
        contribution_metrics: PortfolioContributionMetrics | None = None,
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

        df_positions = pd.DataFrame(
            {
                "simbolo": ["GGAL"],
                "mercado": ["bcba"],
                "cantidad": [10],
                "costo_unitario": [100.0],
            }
        )
        all_symbols = all_symbols or ["GGAL"]
        available_types = list(available_types or IOL_ASSET_TYPES)

        monkeypatch.setattr(
            portfolio_mod,
            "load_portfolio_data",
            lambda cli, svc: (
                df_positions.copy(),
                list(all_symbols),
                list(available_types),
            ),
        )

        controls = Controls(
            refresh_secs=30,
            hide_cash=False,
            show_usd=False,
            order_by="valor_actual",
            desc=True,
            top_n=10,
            selected_syms=list(all_symbols),
            selected_types=list(available_types),
            symbol_query="",
        )
        monkeypatch.setattr(portfolio_mod, "render_sidebar", lambda *a, **k: controls)

        if df_view is None:
            df_view = pd.DataFrame(
                {"simbolo": ["GGAL"], "valor_actual": [1200.0], "tipo": [IOL_ASSET_TYPES[1]]}
            )

        if contribution_metrics is None:
            contribution_metrics = PortfolioContributionMetrics.empty()
        if totals is None:
            totals = PortfolioTotals(0.0, 0.0, 0.0, float("nan"), 0.0)

        def _fake_snapshot(*_args, **_kwargs):
            return PortfolioViewSnapshot(
                df_view=df_view,
                totals=totals,
                apply_elapsed=0.0,
                totals_elapsed=0.0,
                generated_at=0.0,
                historical_total=pd.DataFrame(),
                contribution_metrics=contribution_metrics,
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

        # risk analysis now lives in a separate module loaded dynamically; provide stub
        monkeypatch.setattr(portfolio_mod, "render_advanced_analysis", advanced)
        monkeypatch.setattr(
            portfolio_mod,
            "_load_risk_module",
            lambda: SimpleNamespace(render_risk_analysis=lambda *a, **k: risk(*a, **k)),
        )
        monkeypatch.setattr(portfolio_mod, "render_fundamental_analysis", fundamental)
        monkeypatch.setattr(portfolio_mod, "render_technical_badge", technical_badge)

        def _lazy_prompt_stub(
            block: dict[str, Any],
            *,
            button_label: str,
            key: str,
            dataset_token: str,
            fallback_key: str | None = None,
            force_ready: bool = False,
            **_: Any,
        ) -> bool:
            session_key = fallback_key or key
            current_value = bool(fake_st.session_state.get(session_key, False))
            fake_st.checkbox(button_label, key=session_key, value=current_value)
            ready = bool(fake_st.session_state.get(session_key))
            if ready:
                block["status"] = "loaded"
                block.setdefault("dataset_hash", dataset_token)
                block.setdefault("triggered_at", 0.0)
                block.setdefault("loaded_at", 0.0)
            return ready

        monkeypatch.setattr(portfolio_mod, "_prompt_lazy_block", _lazy_prompt_stub)

        def _record_lazy_stub(
            component: str,
            elapsed_ms: float,
            dataset_token: str | None,
            *,
            mount_latency_ms: float | None = None,
        ) -> None:
            payload = {
                "lazy_loaded_component": component,
                "lazy_load_ms": max(float(elapsed_ms), 0.0),
            }
            if mount_latency_ms is not None:
                payload["visual_mount_latency_ms"] = max(float(mount_latency_ms), 0.0)
            portfolio_mod.log_default_telemetry(
                phase="portfolio.lazy_component",
                elapsed_s=max(float(elapsed_ms), 0.0) / 1000.0,
                dataset_hash=str(dataset_token or "none"),
                extra=payload,
            )

        monkeypatch.setattr(portfolio_mod, "_record_lazy_component_load", _record_lazy_stub)

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
        total_cash_ars=totals_raw.get("total_cash_ars", 0.0) or 0.0,
        total_cash_usd=totals_raw.get("total_cash_usd", 0.0) or 0.0,
        total_cash_combined=totals_raw.get("total_cash_combined"),
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
        DummyCtx(),
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
        DummyCtx(),
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


def test_render_portfolio_section_applies_tab_badges_when_flags_active(
    _portfolio_setup,
) -> None:
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
        DummyCtx(),
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


def test_portfolio_ui_respects_raw_iol_asset_types(
    _portfolio_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_st = FakeStreamlit(radio_sequence=[0])
    captured: dict[str, Any] = {}

    def _capture_csv(snapshot, **_kwargs):
        captured["csv"] = snapshot
        return b"csv"

    def _capture_excel(snapshot, **_kwargs):
        captured["excel"] = snapshot
        return b"xlsx"

    monkeypatch.setattr("ui.export.create_csv_bundle", _capture_csv)
    monkeypatch.setattr("ui.export.create_excel_workbook", _capture_excel)

    symbols = ["CED1", "ACC1", "BON1", "LET1", "FCI1"]
    df_view = pd.DataFrame(
        {
            "simbolo": symbols,
            "mercado": ["bcba", "bcba", "bcba", "bcba", "bcba"],
            "cantidad": [5, 10, 3, 7, 2],
            "valor_actual": [1500.0, 2000.0, 900.0, 600.0, 400.0],
            "costo": [1200.0, 1500.0, 800.0, 550.0, 350.0],
            "pl": [300.0, 500.0, 100.0, 50.0, 50.0],
            "pl_%": [20.0, 33.33, 12.5, 9.09, 14.29],
            "pl_d": [15.0, 20.0, 5.0, 3.0, 2.0],
            "tipo": IOL_ASSET_TYPES,
        }
    )

    contributions = PortfolioContributionMetrics(
        by_symbol=pd.DataFrame(
            {
                "simbolo": symbols,
                "tipo": IOL_ASSET_TYPES,
                "valor_actual": df_view["valor_actual"],
                "costo": df_view["costo"],
                "pl": df_view["pl"],
                "pl_d": df_view["pl_d"],
                "valor_actual_pct": [40.0, 35.0, 10.0, 8.0, 7.0],
                "pl_pct": [45.0, 30.0, 10.0, 8.0, 7.0],
            }
        ),
        by_type=pd.DataFrame(
            {
                "tipo": IOL_ASSET_TYPES,
                "valor_actual": df_view["valor_actual"],
                "costo": df_view["costo"],
                "pl": df_view["pl"],
                "pl_d": df_view["pl_d"],
                "valor_actual_pct": [40.0, 35.0, 10.0, 8.0, 7.0],
                "pl_pct": [45.0, 30.0, 10.0, 8.0, 7.0],
            }
        ),
    )

    totals = PortfolioTotals(4400.0, 3400.0, 1000.0, 29.41, 0.0)

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
        df_view=df_view,
        all_symbols=symbols,
        available_types=IOL_ASSET_TYPES,
        totals=totals,
        contribution_metrics=contributions,
    )

    sidebar_controls: list[Controls] = []

    def _sidebar_stub(all_symbols_arg, available_types_arg):
        fake_st.multiselect(
            "Filtrar por símbolo",
            all_symbols_arg,
            default=list(all_symbols_arg),
        )
        fake_st.multiselect(
            "Filtrar por tipo",
            available_types_arg,
            default=list(available_types_arg),
        )
        controls_obj = Controls(
            refresh_secs=30,
            hide_cash=False,
            show_usd=False,
            order_by="valor_actual",
            desc=True,
            top_n=10,
            selected_syms=list(all_symbols_arg),
            selected_types=list(available_types_arg),
            symbol_query="",
        )
        sidebar_controls.append(controls_obj)
        return controls_obj

    monkeypatch.setattr(portfolio_mod, "render_sidebar", _sidebar_stub)

    render_portfolio_section(
        DummyCtx(),
        cli=object(),
        fx_rates={"ccl": 1000.0},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    controls_used = sidebar_controls[-1]

    charts_mod.st = fake_st
    tables_mod.st = fake_st
    fake_st.text_input = lambda *a, **k: ""
    fake_st.column_config = type(
        "_ColumnConfig",
        (),
        {
            "Column": lambda *a, **k: None,
            "NumberColumn": lambda *a, **k: None,
            "LineChartColumn": lambda *a, **k: None,
        },
    )
    charts_mod.render_table(
        df_view,
        controls_used,
        ccl_rate=1000.0,
    )
    export_mod.st = fake_st
    export_mod.render_portfolio_exports(
        snapshot=None,
        df_view=df_view,
        totals=totals,
        historical_total=pd.DataFrame(),
        contribution_metrics=contributions,
    )

    type_filter = next(call for call in fake_st.multiselect_calls if call["label"] == "Filtrar por tipo")
    assert type_filter["options"] == IOL_ASSET_TYPES
    assert type_filter["rendered"] == IOL_ASSET_TYPES

    table_df: pd.DataFrame | None = None
    for data, _ in fake_st.dataframes:
        table_data = getattr(data, "data", None)
        if isinstance(table_data, pd.DataFrame) and "Tipo" in table_data.columns:
            table_df = table_data
            break

    assert table_df is not None, "La tabla principal debe registrar datos para validar tipos"
    assert sorted(table_df["Tipo"].unique()) == sorted(IOL_ASSET_TYPES)

    assert "csv" in captured
    csv_snapshot = captured["csv"]
    assert sorted(csv_snapshot.positions["tipo"].unique()) == sorted(IOL_ASSET_TYPES)
    assert sorted(csv_snapshot.contributions_by_symbol["tipo"].unique()) == sorted(IOL_ASSET_TYPES)
    assert sorted(csv_snapshot.contributions_by_type["tipo"].unique()) == sorted(IOL_ASSET_TYPES)

    for tipo in csv_snapshot.positions["tipo"].unique():
        assert tipo in IOL_ASSET_TYPES

    basic.assert_called()
    advanced.assert_not_called()
    risk.assert_not_called()
    fundamental.assert_not_called()
    technical_badge.assert_not_called()


def test_render_portfolio_section_renders_symbol_selector_for_favorites(
    _portfolio_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_st = FakeStreamlit(
        radio_sequence=[4],
        selectbox_defaults={
            "Seleccioná un símbolo (CEDEAR / ETF)": "GGAL",
            "Período": "6mo",
            "Intervalo": "1d",
        },
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
    monkeypatch.setattr(
        portfolio_mod,
        "plot_technical_analysis_chart",
        lambda df, fast, slow: {"df": df, "fast": fast, "slow": slow},
    )

    render_portfolio_section(
        DummyCtx(),
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
            "Tipo de activo a incluir": "Acciones",
        },
    )
    risk_mod.st = fake_st
    fake_st.session_state["selected_asset_types"] = ["Acciones"]

    tab_labels: list[list[str]] = []

    def _capture_tabs(labels: list[str]):
        tab_labels.append(list(labels))
        return [_ContextManager(fake_st) for _ in labels]

    fake_st.tabs = _capture_tabs  # type: ignore[assignment]

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
            "tipo": ["Acciones", "Acciones", "Bono"],
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
    monkeypatch.setattr(
        risk_mod,
        "beta",
        lambda returns, bench: 0.75 if not returns.empty else float("nan"),
    )

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
    pm.render_risk_analysis(
        df_view,
        tasvc,
        favorites=FavoriteSymbols({}),
        available_types=IOL_ASSET_TYPES,
    )

    tags = [call["fig"].tag for call in fake_st.plot_calls if hasattr(call["fig"], "tag")]
    expected_tags = {
        "volatility_dist",
        "portfolio_drawdown",
        "rolling_corr",
        "returns_hist",
    }
    assert expected_tags.issubset(tags)
    assert any("CVaR" in label for label, *_ in fake_st.metrics)
    labels = [call["label"] for call in fake_st.selectbox_calls]
    assert "Calcular correlación sobre el último período:" in labels
    filtered_calls = [call for call in history_calls if call["simbolos"] and not call["simbolos"][0].startswith("^")]
    assert filtered_calls, "Expected history calls for portfolio symbols"
    assert all(set(call["simbolos"]).issubset({"A1", "A2"}) for call in filtered_calls), filtered_calls
    assert all("B" not in call["simbolos"] for call in filtered_calls)


def test_risk_analysis_ui_handles_missing_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    fake_st.session_state["selected_asset_types"] = ["Acciones", "Bono"]

    tab_labels: list[list[str]] = []

    def _capture_tabs(labels: list[str]):
        tab_labels.append(list(labels))
        return [_ContextManager(fake_st) for _ in labels]

    fake_st.tabs = _capture_tabs  # type: ignore[assignment]

    monkeypatch.setattr(risk_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "render_favorite_toggle", lambda *a, **k: None)

    df_view = pd.DataFrame(
        {
            "simbolo": ["A", "B"],
            "valor_actual": [100.0, 80.0],
            "tipo": ["Acciones", "Bono"],
        }
    )

    def single_history(simbolos=None, period="1y"):
        if simbolos and simbolos[0] == "^GSPC":
            return pd.DataFrame({"^GSPC": [100.0, 101.0, 102.0]})
        # Solo la serie de "A" está disponible, "B" faltará para forzar la alerta.
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
        available_types=IOL_ASSET_TYPES,
    )

    assert any("volatilidad" in msg.lower() for msg in fake_st.warnings)
    assert tab_labels, "Se esperaban pestañas de tipo en el panel de riesgo"
    assert all(label in IOL_ASSET_TYPES for label in tab_labels[0])


def test_risk_analysis_warns_when_selected_type_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import controllers.portfolio.risk as risk_mod

    fake_st = FakeStreamlit(
        radio_sequence=[],
        selectbox_defaults={"Tipo de activo a incluir": "Bono"},
    )
    risk_mod.st = fake_st
    fake_st.session_state["selected_asset_types"] = ["Bono"]

    monkeypatch.setattr(risk_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "render_favorite_toggle", lambda *a, **k: None)

    df_view = pd.DataFrame({"simbolo": ["A"], "valor_actual": [100.0], "tipo": ["Acciones"]})

    def _should_not_run(*_, **__):
        raise AssertionError("No debería consultarse histórico cuando no hay datos para el tipo")

    tasvc = SimpleNamespace(portfolio_history=_should_not_run)

    risk_mod.render_risk_analysis(
        df_view,
        tasvc,
        favorites=FavoriteSymbols({}),
        available_types=IOL_ASSET_TYPES,
    )

    assert any("No hay datos para los tipos seleccionados" in msg for msg in fake_st.warnings)


def test_render_advanced_analysis_controls_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            "tipo": ["Acciones"],
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


def test_render_basic_section_renders_heatmap_without_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            "tipo": ["Acciones", "Bono"],
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
                "tipo": ["Acciones", "Bono"],
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
                "tipo": ["Acciones", "Bono"],
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
    assert "portfolio_timeline" not in plotted_keys
    assert {"portfolio_contribution_heatmap", "portfolio_contribution_table"}.issubset(plotted_keys)
    assert not any("históricos" in msg for msg in fake_st.warnings)


def test_render_basic_section_handles_missing_analytics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(charts_mod, "get_persistent_favorites", lambda: favorites_stub)
    monkeypatch.setattr(charts_mod, "render_favorite_toggle", lambda *a, **k: None)
