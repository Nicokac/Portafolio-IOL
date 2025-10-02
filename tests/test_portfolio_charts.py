from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from controllers.portfolio.charts import generate_basic_charts
from services.portfolio_view import PortfolioContributionMetrics
from ui.charts import plot_portfolio_timeline, plot_contribution_heatmap


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


def test_plot_portfolio_timeline_handles_history():
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="D"),
            "total_value": [1000.0, 1050.0, 990.0, 1100.0],
            "total_cost": [900.0, 900.0, 900.0, 900.0],
        }
    )

    fig = plot_portfolio_timeline(history)

    assert isinstance(fig, go.Figure)
    assert fig.data, "Expected timeline figure to contain traces"


def test_plot_contribution_heatmap_from_contribution_metrics():
    metrics = PortfolioContributionMetrics(
        by_symbol=pd.DataFrame(
            {
                "tipo": ["ACCION", "ACCION", "BONO"],
                "simbolo": ["GGAL", "YPFD", "AL30"],
                "valor_actual": [1000.0, 500.0, 300.0],
                "costo": [800.0, 600.0, 250.0],
                "pl": [200.0, -100.0, 50.0],
                "pl_d": [10.0, -5.0, 1.0],
                "valor_actual_pct": [50.0, 25.0, 25.0],
                "pl_pct": [57.14, -28.57, 14.29],
            }
        ),
        by_type=pd.DataFrame(
            {
                "tipo": ["ACCION", "BONO"],
                "valor_actual": [1500.0, 300.0],
                "costo": [1400.0, 250.0],
                "pl": [100.0, 50.0],
                "pl_d": [5.0, 1.0],
                "valor_actual_pct": [83.33, 16.67],
                "pl_pct": [66.67, 33.33],
            }
        ),
    )

    fig = plot_contribution_heatmap(metrics.by_symbol)

    assert isinstance(fig, go.Figure)
    assert fig.data, "Heatmap should include data traces"


def test_plot_contribution_heatmap_handles_empty_df():
    empty_metrics = PortfolioContributionMetrics.empty()

    assert plot_portfolio_timeline(pd.DataFrame()) is None
    assert plot_contribution_heatmap(empty_metrics.by_symbol) is None
