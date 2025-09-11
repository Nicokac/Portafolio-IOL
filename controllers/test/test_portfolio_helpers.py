import pandas as pd
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from controllers import portfolio as pm
from domain.models import Controls


class DummyCtx:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def test_load_portfolio_data(monkeypatch):
    payload = {"activos": [{"simbolo": "AL30", "mercado": "BCBA"}]}
    monkeypatch.setattr(pm, "fetch_portfolio", lambda cli: payload)
    monkeypatch.setattr(pm.st, "spinner", lambda msg: DummyCtx())
    monkeypatch.setattr(pm.st, "warning", lambda *a, **k: None)
    monkeypatch.setattr(pm.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(pm.st, "error", lambda *a, **k: None)
    monkeypatch.setattr(pm.st, "dataframe", lambda *a, **k: None)
    monkeypatch.setattr(pm.st, "stop", lambda: None)

    class DummyPSvc:
        def normalize_positions(self, payload):
            return pd.DataFrame(payload["activos"])

        def classify_asset_cached(self, sym):
            return {"AL30": "Bono"}.get(sym)

    df_pos, syms, types = pm._load_portfolio_data(None, DummyPSvc())
    assert list(df_pos["simbolo"]) == ["AL30"]
    assert syms == ["AL30"]
    assert types == ["Bono"]


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
    monkeypatch.setattr(pm, "fetch_quotes_bulk", lambda cli, pairs: quotes)
    monkeypatch.setattr(pm.time, "time", lambda: 1)
    pm.st.session_state = {}

    class DummyPSvc:
        def calc_rows(self, quote_fn, df, exclude_syms=None):
            df = df.copy()
            df["valor_actual"] = 100
            return df

        def classify_asset_cached(self, sym):
            return {"AL30": "Bono", "GOOG": "Accion"}.get(sym)

    df_view = pm._apply_filters(df_pos, controls, None, DummyPSvc())
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
    monkeypatch.setattr(pm, "plot_pl_topn", lambda df, n: "topn")
    monkeypatch.setattr(pm, "plot_donut_tipo", lambda df: "donut")
    monkeypatch.setattr(pm, "plot_dist_por_tipo", lambda df: "dist")
    monkeypatch.setattr(pm, "plot_pl_daily_topn", lambda df, n: "daily")

    charts = pm._generate_basic_charts(df, top_n=5)
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

    vol, b, var_95, opt_w, port_ret = pm._compute_risk_metrics(
        returns_df, bench_ret, weights
    )
    assert len(port_ret) == len(returns_df)
    assert vol >= 0
    assert opt_w.sum() == pytest.approx(1, rel=1e-5)


def test_render_basic_section_handles_empty(monkeypatch):
    mock_info = MagicMock()
    monkeypatch.setattr(pm.st, "info", mock_info)
    pm._render_basic_section(pd.DataFrame(), Controls(), None)
    mock_info.assert_called_once_with("No hay datos del portafolio para mostrar.")


def test_render_advanced_analysis_no_columns(monkeypatch):
    df = pd.DataFrame()
    monkeypatch.setattr(pm.st, "subheader", lambda *a, **k: None)
    mock_info = MagicMock()
    monkeypatch.setattr(pm.st, "info", mock_info)
    pm._render_advanced_analysis(df)
    mock_info.assert_called_once_with("No hay columnas disponibles para el gráfico bubble.")


def test_render_risk_analysis_insufficient_symbols(monkeypatch):
    df = pd.DataFrame({"simbolo": ["AL30"]})
    monkeypatch.setattr(pm.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(pm.st, "selectbox", lambda *a, **k: "1y")
    mock_info = MagicMock()
    monkeypatch.setattr(pm.st, "info", mock_info)
    monkeypatch.setattr(pm.st, "spinner", lambda *a, **k: DummyCtx())
    tasvc = SimpleNamespace(portfolio_history=lambda *a, **k: pd.DataFrame())
    pm._render_risk_analysis(df, tasvc)
    mock_info.assert_any_call(
        "Necesitas al menos 2 activos en tu portafolio (después de aplicar filtros) para calcular la correlación."
    )


def test_render_fundamental_analysis_no_symbols(monkeypatch):
    df = pd.DataFrame(columns=["simbolo"])
    monkeypatch.setattr(pm.st, "subheader", lambda *a, **k: None)
    mock_info = MagicMock()
    monkeypatch.setattr(pm.st, "info", mock_info)
    tasvc = SimpleNamespace()
    pm._render_fundamental_analysis(df, tasvc)
    mock_info.assert_called_once_with("No hay símbolos en el portafolio para analizar.")
