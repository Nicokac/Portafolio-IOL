import pandas as pd
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from controllers import portfolio as pm
import controllers.portfolio.load_data as load_mod
import controllers.portfolio.filters as filters_mod
import controllers.portfolio.charts as charts_mod
import controllers.portfolio.risk as risk_mod
import controllers.portfolio.fundamentals as fund_mod
from domain.models import Controls


class DummyCtx:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def test_load_portfolio_data(monkeypatch):
    payload = {"activos": [{"simbolo": "AL30", "mercado": "BCBA"}]}
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
            return {"AL30": "Bono"}.get(sym)

    df_pos, syms, types = pm.load_portfolio_data(None, DummyPSvc())
    assert list(df_pos["simbolo"]) == ["AL30"]
    assert syms == ["AL30"]
    assert types == ["Bono"]

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

    def stop():
        raise StopCalled()

    monkeypatch.setattr(load_mod.st, "stop", stop)
    logger_mock = MagicMock()
    monkeypatch.setattr(load_mod, "logger", logger_mock)

    class DummyPSvc:
        def normalize_positions(self, payload):
            return pd.DataFrame()

    with pytest.raises(StopCalled):
        pm.load_portfolio_data(None, DummyPSvc())

    err_mock.assert_called_once()
    msg = err_mock.call_args[0][0]
    assert msg == "No se pudo cargar el portafolio, intente más tarde"
    assert "detalle interno" not in msg
    logger_mock.error.assert_called_once()


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
    monkeypatch.setattr(charts_mod, "plot_dist_por_tipo", lambda df: "dist")
    monkeypatch.setattr(charts_mod, "plot_pl_daily_topn", lambda df, n: "daily")

    charts = pm.generate_basic_charts(df, top_n=5)
    assert charts == {
        "pl_topn": "topn",
        "donut_tipo": "donut",
        "dist_tipo": "dist",
        "pl_diario": "daily",
    }


def test_compute_risk_metrics():
    returns_df = pd.DataFrame({"A": [0.1, -0.05, 0.02], "B": [0.0, 0.02, -0.01]})
    bench_ret = pd.Series([0.05, 0.01, 0.03])
    weights = pd.Series({"A": 0.5, "B": 0.5})

    expected_port_ret = returns_df.mul(weights, axis=1).sum(axis=1)
    expected_vol = risk_mod.annualized_volatility(expected_port_ret)
    expected_beta = risk_mod.beta(expected_port_ret, bench_ret)
    expected_var = risk_mod.historical_var(expected_port_ret)
    expected_opt_w = risk_mod.markowitz_optimize(returns_df)

    vol, b, var_95, opt_w, port_ret = pm.compute_risk_metrics(
        returns_df, bench_ret, weights
    )

    pd.testing.assert_series_equal(port_ret, expected_port_ret)
    assert vol == pytest.approx(expected_vol)
    assert b == pytest.approx(expected_beta)
    assert var_95 == pytest.approx(expected_var)
    pd.testing.assert_series_equal(opt_w, expected_opt_w)


def test_render_basic_section_handles_empty(monkeypatch):
    mock_info = MagicMock()
    monkeypatch.setattr(charts_mod.st, "info", mock_info)
    pm.render_basic_section(pd.DataFrame(), Controls(), None)
    mock_info.assert_called_once_with("No hay datos del portafolio para mostrar.")


def test_render_advanced_analysis_no_columns(monkeypatch):
    df = pd.DataFrame()
    monkeypatch.setattr(charts_mod.st, "subheader", lambda *a, **k: None)
    mock_info = MagicMock()
    monkeypatch.setattr(charts_mod.st, "info", mock_info)
    pm.render_advanced_analysis(df)
    mock_info.assert_called_once_with("No hay columnas disponibles para el gráfico bubble.")


def test_render_risk_analysis_insufficient_symbols(monkeypatch):
    df = pd.DataFrame({"simbolo": ["AL30"]})
    monkeypatch.setattr(risk_mod.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "selectbox", lambda *a, **k: "1y")
    mock_info = MagicMock()
    monkeypatch.setattr(risk_mod.st, "info", mock_info)
    monkeypatch.setattr(risk_mod.st, "spinner", lambda *a, **k: DummyCtx())
    tasvc = SimpleNamespace(portfolio_history=lambda *a, **k: pd.DataFrame())
    pm.render_risk_analysis(df, tasvc)
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
    tasvc = SimpleNamespace(portfolio_history=lambda *a, **k: pd.DataFrame())
    pm.render_risk_analysis(df, tasvc)
    info_mock.assert_any_call(
        "No se pudieron obtener datos históricos para calcular métricas de riesgo."
    )


def test_render_risk_analysis_valid_data(monkeypatch):
    df = pd.DataFrame({"simbolo": ["A", "B"], "valor_actual": [100, 200]})
    monkeypatch.setattr(risk_mod.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(
        risk_mod.st, "selectbox", lambda label, options, index=0: options[index]
    )
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

    col1 = SimpleNamespace(metric=MagicMock())
    col2 = SimpleNamespace(metric=MagicMock())
    col3 = SimpleNamespace(metric=MagicMock())
    monkeypatch.setattr(risk_mod.st, "columns", lambda n: (col1, col2, col3))

    returns_df = pd.DataFrame({"A": [0.1, 0.2], "B": [0.05, 0.1]})
    bench_ret = pd.Series([0.03, 0.04])

    def fake_compute_returns(df):
        if "A" in df.columns:
            return returns_df
        return pd.DataFrame({"^GSPC": bench_ret})

    monkeypatch.setattr(risk_mod, "compute_returns", fake_compute_returns)

    fake_port_ret = pd.Series([0.01, -0.02])
    opt_w = pd.Series({"A": 0.6, "B": 0.4})
    monkeypatch.setattr(
        risk_mod,
        "compute_risk_metrics",
        lambda r, b, w: (0.2, 1.1, 0.03, opt_w, fake_port_ret),
    )

    monkeypatch.setattr(risk_mod.px, "line", lambda *a, **k: "fig")
    monkeypatch.setattr(
        risk_mod.px,
        "histogram",
        lambda *a, **k: SimpleNamespace(add_vline=lambda *a, **k: None),
    )

    monkeypatch.setattr(
        risk_mod, "monte_carlo_simulation", lambda *a, **k: pd.Series([1, 2])
    )
    monkeypatch.setattr(risk_mod, "apply_stress", lambda *a, **k: 1.05)

    def fake_history(simbolos=None, period=None):
        if simbolos == ["^GSPC"]:
            return pd.DataFrame({"^GSPC": [1, 1.01, 1.02]})
        return pd.DataFrame({"A": [1, 1.1, 1.2], "B": [1, 1.05, 1.1]})

    tasvc = SimpleNamespace(portfolio_history=fake_history)

    pm.render_risk_analysis(df, tasvc)

    assert col1.metric.call_args[0] == (
        "Volatilidad anualizada",
        "20.00%",
    )
    assert col2.metric.call_args[0] == (
        "Beta vs S&P 500",
        "1.10",
    )
    assert col3.metric.call_args[0] == (
        "VaR 5%",
        "3.00%",
    )
    assert len(expander_calls) == 5


def test_render_fundamental_analysis_no_symbols(monkeypatch):
    df = pd.DataFrame(columns=["simbolo"])
    monkeypatch.setattr(fund_mod.st, "subheader", lambda *a, **k: None)
    mock_info = MagicMock()
    monkeypatch.setattr(fund_mod.st, "info", mock_info)
    tasvc = SimpleNamespace()
    pm.render_fundamental_analysis(df, tasvc)
    mock_info.assert_called_once_with("No hay símbolos en el portafolio para analizar.")
