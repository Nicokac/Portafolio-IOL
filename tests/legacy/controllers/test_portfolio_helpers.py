import pandas as pd
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
import plotly.express as px

# NOTE: Helpers legacy mantenidos temporalmente para comparar con la suite
# moderna de portfolio. Una vez validada la cobertura en `tests/controllers/`
# estos escenarios se eliminarán.
from controllers import portfolio as pm
import controllers.portfolio.load_data as load_mod
import controllers.portfolio.filters as filters_mod
import controllers.portfolio.charts as charts_mod
import controllers.portfolio.risk as risk_mod
import controllers.portfolio.fundamentals as fund_mod
from shared.favorite_symbols import FavoriteSymbols
from domain.models import Controls


class DummyCtx:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def test_load_portfolio_data(monkeypatch):
    payload = {
        "activos": [
            {"simbolo": "AL30", "mercado": "BCBA"},
            {"simbolo": "GOOG", "mercado": "NASDAQ"},
            {"simbolo": "AL30", "mercado": "BCBA"},
        ]
    }
    monkeypatch.setattr(load_mod, "fetch_portfolio", lambda cli: payload)
    monkeypatch.setattr(load_mod.st, "spinner", lambda msg: DummyCtx())
    monkeypatch.setattr(load_mod.st, "warning", lambda *a, **k: None)
    monkeypatch.setattr(load_mod.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(load_mod.st, "error", lambda *a, **k: None)
    monkeypatch.setattr(load_mod.st, "dataframe", lambda *a, **k: None)
    monkeypatch.setattr(load_mod.st, "stop", lambda: None)

    class DummyPSvc:
        def normalize_positions(self, payload):
            return pd.DataFrame(payload["activos"])

        def classify_asset_cached(self, sym):
            return {"AL30": "Bono", "GOOG": "Accion"}.get(sym)

    df_pos, syms, types = pm.load_portfolio_data(None, DummyPSvc())
    assert list(df_pos["simbolo"]) == ["AL30", "GOOG", "AL30"]
    assert syms == ["AL30", "GOOG"]
    assert types == ["Accion", "Bono"]

def test_load_portfolio_data_reruns_on_expired_session(monkeypatch):
    monkeypatch.setattr(load_mod, "fetch_portfolio", lambda cli: {})
    monkeypatch.setattr(load_mod.st, "spinner", lambda msg: DummyCtx())
    warn_mock = MagicMock()
    monkeypatch.setattr(load_mod.st, "warning", warn_mock)
    monkeypatch.setattr(load_mod.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(load_mod.st, "error", lambda *a, **k: None)
    monkeypatch.setattr(load_mod.st, "dataframe", lambda *a, **k: None)
    monkeypatch.setattr(load_mod.st, "stop", lambda: None)
    monkeypatch.setattr(load_mod.st, "session_state", {"force_login": True}, raising=False)

    class RerunCalled(Exception):
        pass

    def rerun():
        raise RerunCalled()

    monkeypatch.setattr(load_mod.st, "rerun", rerun)

    class DummyPSvc:
        def normalize_positions(self, payload):
            return pd.DataFrame()

        def classify_asset_cached(self, sym):
            return None

    with pytest.raises(RerunCalled):
        pm.load_portfolio_data(None, DummyPSvc())

    warn_mock.assert_called_once_with(
        "Sesión expirada, por favor vuelva a iniciar sesión"
    )


def test_load_portfolio_data_shows_generic_error(monkeypatch):
    def boom(cli):
        raise ValueError("detalle interno")

    monkeypatch.setattr(load_mod, "fetch_portfolio", boom)
    monkeypatch.setattr(load_mod.st, "spinner", lambda msg: DummyCtx())
    monkeypatch.setattr(load_mod.st, "warning", lambda *a, **k: None)
    monkeypatch.setattr(load_mod.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(load_mod.st, "dataframe", lambda *a, **k: None)

    err_mock = MagicMock()
    monkeypatch.setattr(load_mod.st, "error", err_mock)

    class StopCalled(Exception):
        pass

    stop_mock = MagicMock(side_effect=StopCalled)
    monkeypatch.setattr(load_mod.st, "stop", stop_mock)
    logger_mock = MagicMock()
    monkeypatch.setattr(load_mod, "logger", logger_mock)

    class DummyPSvc:
        def normalize_positions(self, payload):
            return pd.DataFrame()

    with pytest.raises(StopCalled):
        pm.load_portfolio_data(None, DummyPSvc())

    err_mock.assert_called_once()
    stop_mock.assert_called_once()
    msg = err_mock.call_args[0][0]
    assert msg == "No se pudo cargar el portafolio, intente más tarde"
    assert "detalle interno" not in msg
    logger_mock.exception.assert_called_once()


def test_load_portfolio_data_reruns_on_auth_error(monkeypatch):
    payload = {"status": 401}
    monkeypatch.setattr(load_mod, "fetch_portfolio", lambda cli: payload)
    monkeypatch.setattr(load_mod.st, "spinner", lambda msg: DummyCtx())
    warn_mock = MagicMock()
    monkeypatch.setattr(load_mod.st, "warning", warn_mock)
    monkeypatch.setattr(load_mod.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(load_mod.st, "error", lambda *a, **k: None)
    monkeypatch.setattr(load_mod.st, "dataframe", lambda *a, **k: None)
    stop_mock = MagicMock()
    monkeypatch.setattr(load_mod.st, "stop", stop_mock)
    monkeypatch.setattr(load_mod.st, "session_state", {}, raising=False)

    class RerunCalled(Exception):
        pass

    rerun_mock = MagicMock(side_effect=RerunCalled)
    monkeypatch.setattr(load_mod.st, "rerun", rerun_mock)

    class DummyPSvc:
        def normalize_positions(self, payload):
            pytest.fail("normalize_positions should not be called")

        def classify_asset_cached(self, sym):
            return None

    with pytest.raises(RerunCalled):
        pm.load_portfolio_data(None, DummyPSvc())

    warn_mock.assert_called_once_with(
        "Sesión expirada, por favor vuelva a iniciar sesión"
    )
    rerun_mock.assert_called_once()
    stop_mock.assert_not_called()


def test_load_portfolio_data_warns_on_empty_positions(monkeypatch):
    payload = {"activos": []}
    monkeypatch.setattr(load_mod, "fetch_portfolio", lambda cli: payload)
    monkeypatch.setattr(load_mod.st, "spinner", lambda msg: DummyCtx())
    warn_mock = MagicMock()
    monkeypatch.setattr(load_mod.st, "warning", warn_mock)
    monkeypatch.setattr(load_mod.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(load_mod.st, "error", lambda *a, **k: None)
    df_mock = MagicMock()
    monkeypatch.setattr(load_mod.st, "dataframe", df_mock)
    class StopCalled(Exception):
        pass
    stop_mock = MagicMock(side_effect=StopCalled)
    monkeypatch.setattr(load_mod.st, "stop", stop_mock)
    monkeypatch.setattr(load_mod.st, "session_state", {}, raising=False)

    class DummyPSvc:
        def normalize_positions(self, payload):
            return pd.DataFrame(payload.get("activos", []))

        def classify_asset_cached(self, sym):
            return None

    with pytest.raises(StopCalled):
        pm.load_portfolio_data(None, DummyPSvc())

    warn_mock.assert_called_once_with(
        "No se encontraron posiciones o no pudimos mapear la respuesta."
    )
    df_mock.assert_called_once()
    stop_mock.assert_called_once()


def test_apply_filters(monkeypatch):
    df_pos = pd.DataFrame(
        [
            {"simbolo": "AL30", "mercado": "BCBA"},
            {"simbolo": "IOLPORA", "mercado": "BCBA"},
            {"simbolo": "GOOG", "mercado": "NASDAQ"},
        ]
    )
    controls = Controls(
        hide_cash=True,
        selected_syms=["AL30", "GOOG"],
        selected_types=["Bono"],
        symbol_query="AL",
    )

    quotes = {
        ("bcba", "AL30"): {"chg_pct": 1.0},
        ("nasdaq", "GOOG"): {"chg_pct": 2.0},
    }
    monkeypatch.setattr(filters_mod, "fetch_quotes_bulk", lambda cli, pairs: quotes)
    monkeypatch.setattr(filters_mod.time, "time", lambda: 1)
    filters_mod.cache = SimpleNamespace(session_state={})

    class DummyPSvc:
        def calc_rows(self, quote_fn, df, exclude_syms=None):
            df = df.copy()
            df["valor_actual"] = 100
            return df

        def classify_asset_cached(self, sym):
            return {"AL30": "Bono", "GOOG": "Accion"}.get(sym)

    df_view = pm.apply_filters(df_pos, controls, None, DummyPSvc())
    assert list(df_view["simbolo"]) == ["AL30"]
    assert "chg_%" in df_view.columns


def test_generate_basic_charts(monkeypatch):
    df = pd.DataFrame({
        "simbolo": ["AL30"],
        "valor_actual": [100],
        "pl": [10],
        "pl_%": [0.1],
        "pl_d": [5],
        "tipo": ["Bono"],
    })
    monkeypatch.setattr(charts_mod, "plot_pl_topn", lambda df, n: "topn")
    monkeypatch.setattr(charts_mod, "plot_donut_tipo", lambda df: "donut")
    monkeypatch.setattr(charts_mod, "plot_pl_daily_topn", lambda df, n: "daily")

    charts = pm.generate_basic_charts(df, top_n=5)
    assert charts == {
        "pl_topn": "topn",
        "donut_tipo": "donut",
        "pl_diario": "daily",
    }


def test_generate_basic_charts_missing_columns():
    df = pd.DataFrame({"foo": [1]})
    charts = pm.generate_basic_charts(df, top_n=5)
    assert charts == {
        "pl_topn": None,
        "donut_tipo": None,
        "pl_diario": None,
    }


def test_compute_risk_metrics():
    returns_df = pd.DataFrame({"A": [0.1, -0.05, 0.02], "B": [0.0, 0.02, -0.01]})
    bench_ret = pd.Series([0.05, 0.01, 0.03])
    weights = pd.Series({"A": 0.5, "B": 0.5})

    expected_port_ret = returns_df.mul(weights, axis=1).sum(axis=1)
    expected_vol = risk_mod.annualized_volatility(expected_port_ret)
    expected_beta = risk_mod.beta(expected_port_ret, bench_ret)
    expected_var = risk_mod.historical_var(expected_port_ret)
    expected_cvar = risk_mod.expected_shortfall(expected_port_ret)
    expected_opt_w = risk_mod.markowitz_optimize(returns_df)
    expected_asset_vols, expected_asset_drawdowns = risk_mod.asset_risk_breakdown(
        returns_df
    )
    expected_port_drawdown = risk_mod.max_drawdown(expected_port_ret)

    (
        vol,
        b,
        var_95,
        cvar_95,
        opt_w,
        port_ret,
        asset_vols,
        asset_drawdowns,
        port_drawdown,
    ) = pm.compute_risk_metrics(returns_df, bench_ret, weights)

    pd.testing.assert_series_equal(port_ret, expected_port_ret)
    assert vol == pytest.approx(expected_vol)
    assert b == pytest.approx(expected_beta)
    assert var_95 == pytest.approx(expected_var)
    assert cvar_95 == pytest.approx(expected_cvar)
    pd.testing.assert_series_equal(opt_w, expected_opt_w)
    pd.testing.assert_series_equal(asset_vols, expected_asset_vols)
    pd.testing.assert_series_equal(asset_drawdowns, expected_asset_drawdowns)
    assert port_drawdown == pytest.approx(expected_port_drawdown)


def test_render_basic_section_handles_empty(monkeypatch):
    mock_info = MagicMock()
    monkeypatch.setattr(charts_mod.st, "info", mock_info)
    monkeypatch.setattr(charts_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_favorite_toggle", lambda *a, **k: None)
    pm.render_basic_section(pd.DataFrame(), Controls(), None, favorites=FavoriteSymbols({}))
    mock_info.assert_called_once_with("No hay datos del portafolio para mostrar.")


def test_render_basic_section_with_data(monkeypatch):
    df = pd.DataFrame({
        "simbolo": ["AL30"],
        "valor_actual": [100],
        "pl": [10],
        "pl_%": [0.1],
        "pl_d": [5],
        "tipo": ["Bono"],
    })
    mock_totals = MagicMock()
    monkeypatch.setattr(charts_mod, "render_totals", mock_totals)
    monkeypatch.setattr(charts_mod, "render_table", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_favorite_toggle", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_portfolio_exports", lambda *a, **k: None)
    fig = object()
    monkeypatch.setattr(
        charts_mod,
        "generate_basic_charts",
        lambda df, top_n: {
            "pl_topn": fig,
            "donut_tipo": fig,
            "pl_diario": fig,
        },
    )
    plot_mock = MagicMock()
    monkeypatch.setattr(charts_mod.st, "plotly_chart", plot_mock)
    monkeypatch.setattr(charts_mod.st, "columns", lambda n: (DummyCtx(), DummyCtx()))
    monkeypatch.setattr(charts_mod.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod.st, "info", lambda *a, **k: None)

    monkeypatch.setattr(charts_mod.st, "selectbox", lambda *a, **k: "AL30")
    pm.render_basic_section(df, Controls(), None, favorites=FavoriteSymbols({}))

    mock_totals.assert_called_once()
    assert plot_mock.call_count == 3


def test_render_advanced_analysis_no_columns(monkeypatch):
    df = pd.DataFrame()
    monkeypatch.setattr(charts_mod.st, "subheader", lambda *a, **k: None)
    mock_info = MagicMock()
    monkeypatch.setattr(charts_mod.st, "info", mock_info)
    monkeypatch.setattr(charts_mod, "compute_symbol_risk_metrics", lambda *a, **k: pd.DataFrame())

    def _fake_columns(n):
        return tuple(DummyCtx() for _ in range(n))

    monkeypatch.setattr(charts_mod.st, "columns", _fake_columns)
    monkeypatch.setattr(
        charts_mod.st,
        "selectbox",
        lambda label, options, index=0, **_: options[index] if options else None,
    )
    monkeypatch.setattr(charts_mod.st, "checkbox", lambda *a, **k: False)
    monkeypatch.setattr(charts_mod.st, "metric", lambda *a, **k: None)

    pm.render_advanced_analysis(df, tasvc=None)
    mock_info.assert_called_once_with("No hay columnas disponibles para el gráfico bubble.")


def test_render_advanced_analysis_missing_bubble_columns(monkeypatch):
    df = pd.DataFrame({"pl": [1, 2], "pl_%": [0.1, 0.2]})
    monkeypatch.setattr(charts_mod.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "compute_symbol_risk_metrics", lambda *a, **k: pd.DataFrame())

    def _fake_selectbox(label, options, index=0, **_):
        desired = {
            "Período de métricas": options[index],
            "Métrica de riesgo": options[index],
            "Benchmark": options[index],
            "Eje X": "pl",
            "Eje Y": "pl_%",
            "Paleta": "Tema",
            "Escala de color": "RdBu",
        }
        return desired.get(label, options[index] if options else None)

    monkeypatch.setattr(charts_mod.st, "selectbox", _fake_selectbox)
    monkeypatch.setattr(charts_mod.st, "checkbox", lambda *a, **k: False)
    monkeypatch.setattr(
        charts_mod.st,
        "columns",
        lambda n: tuple(DummyCtx() for _ in range(n)),
    )
    info_mock = MagicMock()
    monkeypatch.setattr(charts_mod.st, "info", info_mock)
    monkeypatch.setattr(charts_mod.st, "plotly_chart", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod.st, "expander", lambda *a, **k: DummyCtx())
    monkeypatch.setattr(charts_mod.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "plot_bubble_pl_vs_costo", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "plot_heat_pl_pct", lambda *a, **k: None)

    pm.render_advanced_analysis(df, tasvc=None)

    info_mock.assert_any_call("No hay datos suficientes para el gráfico bubble.")


def test_render_advanced_analysis_palette_and_log(monkeypatch):
    df = pd.DataFrame(
        {
            "simbolo": ["AL30", "GOOG"],
            "tipo": ["Bono", "Accion"],
            "valor_actual": [100, 200],
            "costo": [90, 150],
            "pl": [10, 50],
            "pl_d": [1, 2],
            "pl_%": [0.1, 0.2],
        }
    )
    monkeypatch.setattr(charts_mod.st, "subheader", lambda *a, **k: None)
    metrics = pd.DataFrame(
        {
            "simbolo": ["AL30", "GOOG", "^GSPC"],
            "volatilidad": [0.1, 0.2, 0.15],
            "drawdown": [-0.2, -0.3, -0.25],
            "beta": [0.9, 1.1, 1.0],
            "es_benchmark": [False, False, True],
        }
    )
    monkeypatch.setattr(
        charts_mod,
        "compute_symbol_risk_metrics",
        lambda *a, **k: metrics.copy(),
    )

    def _fake_selectbox(label, options, index=0, **_):
        mapping = {
            "Período de métricas": "1y",
            "Métrica de riesgo": "Volatilidad anualizada",
            "Benchmark": "S&P 500 (^GSPC)",
            "Eje X": "costo",
            "Eje Y": "pl",
            "Paleta": "Plotly",
            "Escala de color": "Viridis",
        }
        return mapping.get(label, options[index] if options else None)

    monkeypatch.setattr(charts_mod.st, "selectbox", _fake_selectbox)
    chk_vals = iter([True, True])
    monkeypatch.setattr(charts_mod.st, "checkbox", lambda *a, **k: next(chk_vals))
    monkeypatch.setattr(
        charts_mod.st,
        "columns",
        lambda n: tuple(DummyCtx() for _ in range(n)),
    )
    monkeypatch.setattr(charts_mod.st, "expander", lambda *a, **k: DummyCtx())
    monkeypatch.setattr(charts_mod.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod.st, "plotly_chart", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod.st, "metric", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod.st, "info", lambda *a, **k: None)
    bubble_mock = MagicMock(return_value="fig1")
    heat_mock = MagicMock(return_value="fig2")
    monkeypatch.setattr(charts_mod, "plot_bubble_pl_vs_costo", bubble_mock)
    monkeypatch.setattr(charts_mod, "plot_heat_pl_pct", heat_mock)

    pm.render_advanced_analysis(df, tasvc=None)

    args, kwargs = bubble_mock.call_args
    assert kwargs["x_axis"] == "costo"
    assert kwargs["y_axis"] == "pl"
    assert kwargs["color_seq"] == px.colors.qualitative.Plotly
    assert kwargs["log_x"] is True
    assert kwargs["log_y"] is True
    assert kwargs["category_col"] == "categoria"
    assert kwargs["benchmark_col"] == "es_benchmark"
    result_df = args[0]
    assert "categoria" in result_df.columns
    assert result_df["categoria"].isin(["Activo", "Benchmark"]).all()
    heat_mock.assert_called_once()


def test_render_risk_analysis_insufficient_symbols(monkeypatch):
    df = pd.DataFrame({"simbolo": ["AL30"]})
    monkeypatch.setattr(risk_mod.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "selectbox", lambda *a, **k: "1y")
    mock_info = MagicMock()
    monkeypatch.setattr(risk_mod.st, "info", mock_info)
    monkeypatch.setattr(risk_mod.st, "spinner", lambda *a, **k: DummyCtx())
    monkeypatch.setattr(risk_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "render_favorite_toggle", lambda *a, **k: None)
    tasvc = SimpleNamespace(portfolio_history=lambda *a, **k: pd.DataFrame())
    pm.render_risk_analysis(df, tasvc, favorites=FavoriteSymbols({}))
    mock_info.assert_any_call(
        "Necesitas al menos 2 activos en tu portafolio (después de aplicar filtros) para calcular la correlación."
    )


def test_render_risk_analysis_empty_history(monkeypatch):
    df = pd.DataFrame({"simbolo": ["A", "B"], "valor_actual": [100, 200]})
    monkeypatch.setattr(risk_mod.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "selectbox", lambda *a, **k: "1y")
    monkeypatch.setattr(risk_mod.st, "spinner", lambda *a, **k: DummyCtx())
    monkeypatch.setattr(risk_mod.st, "warning", lambda *a, **k: None)
    info_mock = MagicMock()
    monkeypatch.setattr(risk_mod.st, "info", info_mock)
    monkeypatch.setattr(risk_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "render_favorite_toggle", lambda *a, **k: None)
    tasvc = SimpleNamespace(portfolio_history=lambda *a, **k: pd.DataFrame())
    pm.render_risk_analysis(df, tasvc, favorites=FavoriteSymbols({}))
    info_mock.assert_any_call(
        "No se pudieron obtener datos históricos para calcular métricas de riesgo."
    )


def test_render_risk_analysis_valid_data(monkeypatch):
    df = pd.DataFrame({"simbolo": ["A", "B"], "valor_actual": [100, 200]})
    monkeypatch.setattr(risk_mod.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "spinner", lambda *a, **k: DummyCtx())
    monkeypatch.setattr(risk_mod.st, "plotly_chart", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "warning", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "bar_chart", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "line_chart", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "write", lambda *a, **k: None)
    monkeypatch.setattr(
        risk_mod.st, "number_input", lambda *args, **kwargs: kwargs.get("value")
    )

    expander_calls = []

    def fake_expander(label):
        expander_calls.append(label)
        return DummyCtx()

    monkeypatch.setattr(risk_mod.st, "expander", fake_expander)

    col_metrics = [SimpleNamespace(metric=MagicMock()) for _ in range(5)]
    vol_col = SimpleNamespace(plotly_chart=MagicMock(), info=MagicMock())
    draw_col = SimpleNamespace(plotly_chart=MagicMock(), info=MagicMock())
    scatter_col = SimpleNamespace(plotly_chart=MagicMock(), info=MagicMock())

    columns_calls = iter(
        [col_metrics, (vol_col, draw_col), (scatter_col,)]
    )

    def fake_columns(n):
        return next(columns_calls)

    monkeypatch.setattr(risk_mod.st, "columns", fake_columns)

    returns_df = pd.DataFrame({"A": [0.1, 0.2], "B": [0.05, 0.1]})
    bench_ret = pd.Series([0.03, 0.04])

    def fake_compute_returns(df):
        if "A" in df.columns:
            return returns_df
        return pd.DataFrame({"^GSPC": bench_ret})

    monkeypatch.setattr(risk_mod, "compute_returns", fake_compute_returns)

    fake_port_ret = pd.Series([0.01, -0.02])
    opt_w = pd.Series({"A": 0.6, "B": 0.4})
    asset_vols = pd.Series({"A": 0.15, "B": 0.1})
    asset_drawdowns = pd.Series({"A": -0.2, "B": -0.1})
    port_drawdown = -0.25
    monkeypatch.setattr(
        risk_mod,
        "compute_risk_metrics",
        lambda r, b, w, var_confidence=0.95: (
            0.2,
            1.1,
            0.03,
            0.04,
            opt_w,
            fake_port_ret,
            asset_vols,
            asset_drawdowns,
            port_drawdown,
        ),
    )

    class DummyFigure:
        def update_layout(self, **kwargs):
            return self

        def update_xaxes(self, **kwargs):
            return self

        def update_yaxes(self, **kwargs):
            return self

        def update_traces(self, **kwargs):
            return self

        def add_vline(self, **kwargs):
            return self

    monkeypatch.setattr(risk_mod.px, "line", lambda *a, **k: DummyFigure())
    monkeypatch.setattr(
        risk_mod.px,
        "histogram",
        lambda *a, **k: DummyFigure(),
    )
    monkeypatch.setattr(risk_mod.px, "bar", lambda *a, **k: DummyFigure())
    monkeypatch.setattr(risk_mod.px, "scatter", lambda *a, **k: DummyFigure())
    monkeypatch.setattr(risk_mod, "drawdown_series", lambda *_: pd.Series([-0.01, -0.05]))

    monkeypatch.setattr(
        risk_mod, "monte_carlo_simulation", lambda *a, **k: pd.Series([1, 2])
    )
    monkeypatch.setattr(risk_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "render_favorite_toggle", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "markdown", lambda *a, **k: None)

    def fake_history(simbolos=None, period=None):
        if simbolos == ["^GSPC"]:
            return pd.DataFrame({"^GSPC": [1, 1.01, 1.02]})
        return pd.DataFrame({"A": [1, 1.1, 1.2], "B": [1, 1.05, 1.1]})

    tasvc = SimpleNamespace(portfolio_history=fake_history)

    select_values = iter(
        [["A", "B"], "1y", "3 meses (63)", "S&P 500 (^GSPC)", "95%"]
    )

    def fake_selectbox(label, options, index=0, **kwargs):
        try:
            val = next(select_values)
        except StopIteration:
            val = options[index]
        if isinstance(val, list):
            return val[0]
        return val

    monkeypatch.setattr(risk_mod.st, "selectbox", fake_selectbox)
    monkeypatch.setattr(
        risk_mod, "rolling_correlations", lambda *a, **k: pd.DataFrame({"A↔B": [0.1, 0.2]})
    )

    pm.render_risk_analysis(df, tasvc, favorites=FavoriteSymbols({}))

    assert col_metrics[0].metric.call_args[0] == (
        "Volatilidad anualizada",
        "20.00%",
    )
    assert col_metrics[1].metric.call_args[0] == (
        "Beta vs S&P 500 (^GSPC)",
        "1.10",
    )
    assert col_metrics[2].metric.call_args[0] == (
        "VaR 5%",
        "3.00%",
    )
    assert col_metrics[3].metric.call_args[0] == (
        "CVaR 5%",
        "4.00%",
    )
    assert col_metrics[4].metric.call_args[0] == (
        "Drawdown máximo",
        "-25.00%",
    )
    vol_col.plotly_chart.assert_called_once()
    draw_col.plotly_chart.assert_called_once()
    scatter_col.plotly_chart.assert_called_once()
    assert len(expander_calls) == 5


def test_render_risk_analysis_insufficient_per_asset_data(monkeypatch):
    df = pd.DataFrame({"simbolo": ["A"], "valor_actual": [100]})
    monkeypatch.setattr(risk_mod.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "spinner", lambda *a, **k: DummyCtx())
    info_messages = []

    def fake_info(message):
        info_messages.append(message)

    monkeypatch.setattr(risk_mod.st, "info", fake_info)
    monkeypatch.setattr(risk_mod.st, "warning", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "bar_chart", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "line_chart", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "write", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "plotly_chart", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "caption", lambda *a, **k: None)

    dummy_col = SimpleNamespace(plotly_chart=MagicMock(), info=fake_info)
    monkeypatch.setattr(
        risk_mod.st,
        "columns",
        lambda n: ([SimpleNamespace(metric=MagicMock()) for _ in range(n)]
        if n == 5
        else (dummy_col, dummy_col) if n == 2 else (dummy_col,)),
    )

    returns_df = pd.DataFrame({"A": [0.01, -0.02]})
    bench_ret = pd.Series([0.0, 0.0])

    def fake_history(simbolos=None, period=None):
        if simbolos and simbolos[0] == "^GSPC":
            return pd.DataFrame({"^GSPC": [1.0, 1.01, 1.02]})
        return pd.DataFrame({"A": [1.0, 1.02, 1.01]})

    tasvc = SimpleNamespace(portfolio_history=fake_history)

    monkeypatch.setattr(risk_mod, "compute_returns", lambda df: returns_df)
    monkeypatch.setattr(
        risk_mod,
        "compute_risk_metrics",
        lambda *a, **k: (
            0.1,
            1.0,
            0.02,
            0.03,
            pd.Series({"A": 1.0}),
            pd.Series([0.01, -0.02]),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            -0.05,
        ),
    )
    monkeypatch.setattr(risk_mod, "drawdown_series", lambda *_: pd.Series(dtype=float))
    monkeypatch.setattr(risk_mod.px, "bar", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "markdown", lambda *a, **k: None)

    select_values = iter([["A"], "1y", "S&P 500 (^GSPC)", "95%"])
    monkeypatch.setattr(
        risk_mod.st,
        "selectbox",
        lambda *a, **k: next(select_values),
    )

    pm.render_risk_analysis(df, tasvc, favorites=FavoriteSymbols({}))

    assert any(
        "volatilidad por activo" in msg.lower()
        for msg in info_messages
    )


def test_render_fundamental_analysis_no_symbols(monkeypatch):
    df = pd.DataFrame(columns=["simbolo"])
    monkeypatch.setattr(fund_mod.st, "subheader", lambda *a, **k: None)
    mock_info = MagicMock()
    monkeypatch.setattr(fund_mod.st, "info", mock_info)
    tasvc = SimpleNamespace()
    monkeypatch.setattr(fund_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(fund_mod, "render_favorite_toggle", lambda *a, **k: None)
    pm.render_fundamental_analysis(df, tasvc, favorites=FavoriteSymbols({}))
    mock_info.assert_called_once_with("No hay símbolos en el portafolio para analizar.")
