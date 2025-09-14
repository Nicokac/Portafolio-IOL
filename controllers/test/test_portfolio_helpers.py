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
    logger_mock.error.assert_called_once()


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

    vol, b, var_95, opt_w, port_ret = pm.compute_risk_metrics(
        returns_df, bench_ret, weights
    )
    assert len(port_ret) == len(returns_df)
    assert vol >= 0
    assert opt_w.sum() == pytest.approx(1, rel=1e-5)


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


def test_render_fundamental_analysis_no_symbols(monkeypatch):
    df = pd.DataFrame(columns=["simbolo"])
    monkeypatch.setattr(fund_mod.st, "subheader", lambda *a, **k: None)
    mock_info = MagicMock()
    monkeypatch.setattr(fund_mod.st, "info", mock_info)
    tasvc = SimpleNamespace()
    pm.render_fundamental_analysis(df, tasvc)
    mock_info.assert_called_once_with("No hay símbolos en el portafolio para analizar.")
