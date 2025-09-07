import pandas as pd
import plotly.express as px

from ui.charts import plot_bubble_pl_vs_costo, plot_heat_pl_pct


def test_bubble_chart_log_axes():
    df = pd.DataFrame(
        {
            "simbolo": ["A", "B"],
            "tipo": ["Acción", "Bono"],
            "valor_actual": [100, 200],
            "costo": [90, 190],
            "pl": [10, 10],
        }
    )
    fig = plot_bubble_pl_vs_costo(
        df,
        x_axis="costo",
        y_axis="pl",
        color_seq=px.colors.qualitative.Plotly,
        log_x=True,
        log_y=True,
    )
    assert fig.layout.xaxis.type == "log"
    assert fig.layout.yaxis.type == "log"


def test_heatmap_color_scale():
    df = pd.DataFrame({"simbolo": ["A", "B"], "pl_%": [1.0, -2.0]})
    fig = plot_heat_pl_pct(df, color_scale="Viridis")
    # First color in viridis scale should match
    first_color = px.colors.sequential.Viridis[0]
    assert fig.layout.coloraxis.colorscale[0][1] == first_color