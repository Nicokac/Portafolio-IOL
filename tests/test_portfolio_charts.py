from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from controllers.portfolio.charts import generate_basic_charts
from ui.charts import plot_bubble_pl_vs_costo


def test_generate_basic_charts_produces_figures():
    df = pd.DataFrame(
        [
            {
                "simbolo": "AAPL",
                "pl": 1200.0,
                "tipo": "CEDEAR",
                "valor_actual": 3500.0,
                "pl_d": 150.0,
                "chg_%": 1.5,
            },
            {
                "simbolo": "TSLA",
                "pl": -200.0,
                "tipo": "BONO",
                "valor_actual": 800.0,
                "pl_d": -20.0,
                "chg_%": -0.3,
            },
        ]
    )

    charts = generate_basic_charts(df, top_n=2)

    assert set(charts.keys()) == {"pl_topn", "donut_tipo", "dist_tipo", "pl_diario"}
    for name, fig in charts.items():
        assert isinstance(fig, go.Figure), f"Expected {name} to be a Plotly figure"


def test_plot_bubble_with_benchmark_dataset():
    df = pd.DataFrame(
        {
            "simbolo": ["AL30", "^GSPC"],
            "valor_actual": [1500.0, 0.0],
            "riesgo": [0.22, 0.18],
            "pl": [120.0, 0.0],
            "categoria": ["Activo", "Benchmark"],
            "es_benchmark": [False, True],
            "tipo": ["Accion", "Benchmark"],
        }
    )

    fig = plot_bubble_pl_vs_costo(
        df,
        x_axis="riesgo",
        y_axis="pl",
        category_col="categoria",
        benchmark_col="es_benchmark",
    )

    assert isinstance(fig, go.Figure)
    names = {trace.name for trace in fig.data}
    assert {"Activo", "Benchmark"}.issubset(names)
    assert fig.layout.shapes
    assert any(getattr(shape, "type", None) == "line" for shape in fig.layout.shapes)
