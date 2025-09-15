from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from types import SimpleNamespace
from unittest.mock import MagicMock


from domain.models import Controls
from controllers.portfolio import charts as charts_mod
from controllers.portfolio import risk as risk_mod
from ui.fx_panels import render_spreads
import ui.fx_panels as fx_panels
import ui.tables as tables_mod


class DummyCtx:
    def __enter__(self):
        return None
    def __exit__(self, *exc):
        return False


def test_render_basic_section_captions(monkeypatch):
    df = pd.DataFrame({"simbolo": ["A"], "valor_actual": [1]})
    controls = Controls()
    monkeypatch.setattr(charts_mod, "render_totals", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_table", lambda *a, **k: None)
    monkeypatch.setattr(
        charts_mod,
        "generate_basic_charts",
        lambda df, top_n: {k: object() for k in ["pl_topn", "donut_tipo", "dist_tipo", "pl_diario"]},
    )
    monkeypatch.setattr(charts_mod.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod.st, "plotly_chart", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod.st, "columns", lambda n: (DummyCtx(), DummyCtx()))
    mock_caption = MagicMock()
    monkeypatch.setattr(charts_mod.st, "caption", mock_caption)
    charts_mod.render_basic_section(df, controls, None)
    captions = [c.args[0] for c in mock_caption.call_args_list]
    expected = [
        "Barras que muestran qué activos ganan o pierden más. Las más altas son las que más afectan tu resultado.",
        "Indica qué porcentaje de tu inversión está en cada tipo de activo para ver si estás diversificando bien.",
        "Compara cuánto dinero tenés en cada categoría de activos. Ayuda a detectar concentraciones.",
        "Muestra las ganancias o pérdidas del día para los activos con mayor movimiento.",
    ]
    for text in expected:
        assert any(text in c for c in captions)


def test_render_risk_analysis_caption(monkeypatch):
    df = pd.DataFrame({"simbolo": ["A", "B"], "valor_actual": [1, 2]})
    tasvc = SimpleNamespace(
        portfolio_history=lambda simbolos, period: pd.DataFrame({s: [1, 1] for s in simbolos})
    )
    monkeypatch.setattr(risk_mod.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "selectbox", lambda *a, **k: "1y")
    monkeypatch.setattr(risk_mod.st, "spinner", lambda *a, **k: DummyCtx())
    monkeypatch.setattr(risk_mod.st, "plotly_chart", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod.st, "warning", lambda *a, **k: None)
    mock_caption = MagicMock()
    monkeypatch.setattr(risk_mod.st, "caption", mock_caption)
    monkeypatch.setattr(risk_mod, "plot_correlation_heatmap", lambda df: object())
    monkeypatch.setattr(risk_mod, "compute_returns", lambda df: pd.DataFrame())
    risk_mod.render_risk_analysis(df, tasvc)
    assert any("heatmap de correlación" in str(c.args[0]) for c in mock_caption.call_args_list)


def test_render_spreads_caption(monkeypatch):
    rates = {"ccl": 100, "oficial": 50, "blue": 90, "mep": 80, "mayorista": 60}
    monkeypatch.setattr(fx_panels.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(fx_panels.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(fx_panels.st, "dataframe", lambda *a, **k: None)
    mock_caption = MagicMock()
    monkeypatch.setattr(fx_panels.st, "caption", mock_caption)
    render_spreads(rates)
    mock_caption.assert_called_once_with(
        "Muestra la diferencia porcentual entre distintas cotizaciones del dólar."
    )


def test_render_table_caption(monkeypatch):
    df = pd.DataFrame(
        {
            "mercado": ["A"],
            "simbolo": ["A"],
            "tipo": ["B"],
            "cantidad": [1],
            "ultimo": [1],
            "valor_actual": [1],
            "costo": [1],
            "pl": [0],
            "pl_%": [0],
            "pl_d": [0],
            "chg_%": [0],
        }
    )
    monkeypatch.setattr(
        tables_mod,
        "get_active_palette",
        lambda: SimpleNamespace(bg="", text="", highlight_bg="", highlight_text=""),
    )
    monkeypatch.setattr(tables_mod, "download_csv", lambda *a, **k: None)
    monkeypatch.setattr(tables_mod.st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(tables_mod.st, "text_input", lambda *a, **k: "")
    monkeypatch.setattr(tables_mod.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(tables_mod.st, "dataframe", lambda *a, **k: None)
    monkeypatch.setattr(tables_mod.st, "number_input", lambda *a, **k: 20)
    mock_caption = MagicMock()
    monkeypatch.setattr(tables_mod.st, "caption", mock_caption)
    monkeypatch.setattr(tables_mod.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(tables_mod.st, "session_state", {})
    monkeypatch.setattr(
        tables_mod.st,
        "column_config",
        SimpleNamespace(
            Column=lambda *a, **k: None,
            NumberColumn=lambda *a, **k: None,
            LineChartColumn=lambda *a, **k: None,
        ),
    )
    tables_mod.render_table(df, order_by="valor_actual", desc=False)
    mock_caption.assert_called_once_with(
        "Tabla con todas tus posiciones actuales. Te ayuda a ver cuánto tenés en cada activo y cómo viene rindiendo."
    )
