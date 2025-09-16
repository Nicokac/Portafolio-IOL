from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from controllers.portfolio.charts import generate_basic_charts


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
