import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from ui import charts


def _minimal_df_pl_topn():
    return pd.DataFrame({
        "simbolo": ["AAA"],
        "pl": [1.0],
        "tipo": ["stock"],
    })


def _minimal_df_valor_tipo():
    return pd.DataFrame({
        "valor_actual": [100.0],
        "tipo": ["stock"],
    })


def _minimal_df_heat():
    return pd.DataFrame({
        "simbolo": ["AAA"],
        "pl_%": [5.0],
    })


def _minimal_df_bubble():
    return pd.DataFrame({
        "simbolo": ["AAA", "BBB"],
        "tipo": ["A", "B"],
        "valor_actual": [100, 200],
        "costo": [10, 20],
        "pl": [5, 15],
    })


def _minimal_df_pl_daily():
    return pd.DataFrame({
        "simbolo": ["AAA"],
        "pl_d": [10],
        "pld_%": [1.0],
        "tipo": ["stock"],
    })


def _minimal_df_pl_vs_total():
    return pd.DataFrame({
        "simbolo": ["AAA"],
        "pl": [100],
        "pl_d": [5],
        "tipo": ["stock"],
    })


# plot_pl_topn -------------------------------------------------------------

def test_plot_pl_topn_minimal_df_returns_figure():
    fig = charts.plot_pl_topn(_minimal_df_pl_topn())
    assert isinstance(fig, go.Figure)


def test_plot_pl_topn_missing_columns_or_empty_returns_none():
    assert charts.plot_pl_topn(pd.DataFrame()) is None
    df = _minimal_df_pl_topn().drop(columns=["pl"])
    assert charts.plot_pl_topn(df) is None


# plot_donut_tipo ----------------------------------------------------------

def test_plot_donut_tipo_minimal_df_returns_figure():
    fig = charts.plot_donut_tipo(_minimal_df_valor_tipo())
    assert isinstance(fig, go.Figure)


def test_plot_donut_tipo_missing_columns_or_empty_returns_none():
    assert charts.plot_donut_tipo(pd.DataFrame()) is None
    df = _minimal_df_valor_tipo().drop(columns=["valor_actual"])
    assert charts.plot_donut_tipo(df) is None


# plot_dist_por_tipo -------------------------------------------------------

def test_plot_dist_por_tipo_minimal_df_returns_figure():
    fig = charts.plot_dist_por_tipo(_minimal_df_valor_tipo())
    assert isinstance(fig, go.Figure)


def test_plot_dist_por_tipo_missing_columns_or_empty_returns_none():
    assert charts.plot_dist_por_tipo(pd.DataFrame()) is None
    df = _minimal_df_valor_tipo().drop(columns=["valor_actual"])
    assert charts.plot_dist_por_tipo(df) is None


# plot_bubble_pl_vs_costo --------------------------------------------------

def test_plot_bubble_pl_vs_costo_log_and_color_seq():
    df = _minimal_df_bubble()
    colors = ["#111111", "#222222"]
    fig = charts.plot_bubble_pl_vs_costo(
        df,
        x_axis="costo",
        y_axis="pl",
        color_seq=colors,
        log_x=True,
        log_y=True,
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.type == "log"
    assert fig.layout.yaxis.type == "log"
    # Each tipo becomes a separate trace with corresponding color
    assert fig.data[0].marker.color == colors[0]
    assert fig.data[1].marker.color == colors[1]


def test_plot_bubble_pl_vs_costo_missing_columns_or_empty_returns_none():
    assert charts.plot_bubble_pl_vs_costo(pd.DataFrame(), "costo", "pl") is None
    df = _minimal_df_bubble().drop(columns=["pl"])
    assert charts.plot_bubble_pl_vs_costo(df, "costo", "pl") is None


# plot_heat_pl_pct ---------------------------------------------------------

def test_plot_heat_pl_pct_custom_scale():
    df = _minimal_df_heat()
    fig = charts.plot_heat_pl_pct(df, color_scale="Viridis")
    assert isinstance(fig, go.Figure)
    assert fig.layout.coloraxis.colorscale[0][1] == px.colors.sequential.Viridis[0]


def test_plot_heat_pl_pct_missing_columns_or_empty_returns_none():
    assert charts.plot_heat_pl_pct(pd.DataFrame()) is None
    df = _minimal_df_heat().drop(columns=["pl_%"])
    assert charts.plot_heat_pl_pct(df) is None


# plot_pl_daily_topn -------------------------------------------------------

def test_plot_pl_daily_topn_minimal_df_returns_figure():
    fig = charts.plot_pl_daily_topn(_minimal_df_pl_daily())
    assert isinstance(fig, go.Figure)


def test_plot_pl_daily_topn_missing_columns_or_empty_returns_none():
    assert charts.plot_pl_daily_topn(pd.DataFrame()) is None
    df = _minimal_df_pl_daily().drop(columns=["pl_d"])
    assert charts.plot_pl_daily_topn(df) is None


# plot_pl_daily_vs_total ---------------------------------------------------

def test_plot_pl_daily_vs_total_minimal_df_returns_figure():
    fig = charts.plot_pl_daily_vs_total(_minimal_df_pl_vs_total())
    assert isinstance(fig, go.Figure)


def test_plot_pl_daily_vs_total_missing_columns_or_empty_returns_none():
    assert charts.plot_pl_daily_vs_total(pd.DataFrame()) is None
    df = _minimal_df_pl_vs_total().drop(columns=["pl"])
    assert charts.plot_pl_daily_vs_total(df) is None
