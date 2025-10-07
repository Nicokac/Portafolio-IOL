# ui/charts.py
from __future__ import annotations
import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .palette import get_active_palette

logger = logging.getLogger(__name__)

# =========================
# Tema / colores unificados
# =========================

FONT_FAMILY = "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"

# Mostrar/ocultar títulos de ejes desde un flag único
SHOW_AXIS_TITLES = True

# Paleta estable por símbolo
_SYMBOL_PALETTE = (
    px.colors.qualitative.Set2
    + px.colors.qualitative.Pastel
    + px.colors.qualitative.Light24
)

def _rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba string with given alpha."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

def _symbol_color_map(symbols: list[str]) -> dict[str, str]:
    return {s: _SYMBOL_PALETTE[i % len(_SYMBOL_PALETTE)] for i, s in enumerate(symbols)}

def _color_discrete_map(df: pd.DataFrame, tipo_col: str = "tipo"):
    pal = get_active_palette()
    tipos = [t for t in df[tipo_col].dropna().unique().tolist()]
    return {t: pal.categories.get(t, pal.accent) for t in tipos}

def _si(n: float) -> str:
    try:
        return f"{n:,.0f}".replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception:
        return str(n)

def _apply_layout(fig: go.Figure, title: str | None = None, *, show_legend=True, y0_line=False):
    pal = get_active_palette()
    fig.update_layout(
        title=({"text": title, "x": 0.0, "xanchor": "left"} if title else None),
        font=dict(family=FONT_FAMILY, color=pal.text, size=14),
        paper_bgcolor=pal.bg,
        plot_bgcolor=pal.plot_bg,
        margin=dict(l=16, r=12, t=48 if title else 24, b=32),
        legend_title_text="",
        showlegend=show_legend,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=pal.grid)
    fig.update_yaxes(showgrid=True, gridcolor=pal.grid, zeroline=y0_line, zerolinecolor=pal.grid)
    return fig

# =========================
# Gráficos – Portafolio
# =========================

def plot_pl_topn(df: pd.DataFrame, n: int = 20):
    if df is None or df.empty or "pl" not in df.columns:
        return None
    d = df.dropna(subset=["pl"]).sort_values("pl", ascending=False).head(n)
    if d.empty:
        return None
    order = d["simbolo"].astype(str).tolist()
    sym_map = _symbol_color_map(order)
    fig = px.bar(
        d, x="simbolo", y="pl",
        color="simbolo",
        hover_data={"tipo": True, "pl": ":,.0f"},
        color_discrete_map=sym_map,
        category_orders={"simbolo": order},
    )
    fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside", cliponaxis=False)
    if SHOW_AXIS_TITLES:
        fig.update_xaxes(title="símbolo")
        fig.update_yaxes(title="P/L", tickformat=",")
    else:
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None, tickformat=",")
    return _apply_layout(fig, show_legend=False)

def plot_donut_tipo(df: pd.DataFrame):
    if df is None or df.empty or "valor_actual" not in df.columns or "tipo" not in df.columns:
        return None
    d = (df.dropna(subset=["valor_actual"])
           .groupby("tipo", dropna=True)["valor_actual"].sum().reset_index())
    if d.empty:
        return None
    fig = px.pie(
        d, names="tipo", values="valor_actual", hole=0.60,
        color="tipo", color_discrete_map=_color_discrete_map(d, "tipo"),
    )
    fig.update_traces(
        textinfo="percent",
        hovertemplate="%{label}: %{value:,.0f}<extra></extra>",
        pull=[0.02]*len(d),
    )
    total = _si(float(d["valor_actual"].sum()))
    fig.add_annotation(text=f"<b>Total</b><br>{total}", showarrow=False, font=dict(size=14), x=0.5, y=0.5)
    return _apply_layout(fig, show_legend=True)

def plot_dist_por_tipo(df: pd.DataFrame):
    if df is None or df.empty or "valor_actual" not in df.columns or "tipo" not in df.columns:
        return None
    d = (df.dropna(subset=["valor_actual"])
           .groupby("tipo", dropna=True)["valor_actual"].sum().reset_index())
    if d.empty:
        return None
    order = d.sort_values("valor_actual", ascending=False)["tipo"].tolist()
    fig = px.bar(
        d, x="tipo", y="valor_actual",
        color="tipo", color_discrete_map=_color_discrete_map(d, "tipo"),
        hover_data={"valor_actual": ":,.0f"},
        category_orders={"tipo": order},
    )
    fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside", cliponaxis=False)
    if SHOW_AXIS_TITLES:
        fig.update_xaxes(title="tipo")
        fig.update_yaxes(title="Valorizado", tickformat=",")
    else:
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None, tickformat=",")
    return _apply_layout(fig, show_legend=False)


def plot_portfolio_timeline(history_df: pd.DataFrame | None):
    """Plot the historical evolution of the portfolio totals."""

    if history_df is None or history_df.empty:
        return None

    df = history_df.copy()
    time_col = "timestamp"
    if time_col not in df.columns:
        return None

    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], unit="s", errors="coerce")

    df = df.dropna(subset=[time_col])
    if df.empty:
        return None

    value_cols = [
        col
        for col in ["total_value", "total_cost", "total_pl"]
        if col in df.columns
    ]
    if not value_cols:
        return None

    df = df.sort_values(time_col)
    melted = df.melt(
        id_vars=[time_col],
        value_vars=value_cols,
        var_name="metric",
        value_name="value",
    )
    melted = melted.dropna(subset=["value"])
    if melted.empty:
        return None

    metrics = melted["metric"].astype(str).unique().tolist()
    color_map = _symbol_color_map(metrics)

    fig = px.line(
        melted,
        x=time_col,
        y="value",
        color="metric",
        color_discrete_map=color_map,
        markers=True,
    )

    if SHOW_AXIS_TITLES:
        fig.update_xaxes(title="Fecha")
        fig.update_yaxes(title="Valor", tickformat=",")
    else:
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None, tickformat=",")

    fig.update_traces(mode="lines+markers")
    return _apply_layout(fig, show_legend=True)


def plot_contribution_heatmap(by_symbol: pd.DataFrame | None, *, value_col: str = "valor_actual_pct"):
    """Render a heatmap of contributions grouped by type and symbol."""

    if by_symbol is None or by_symbol.empty:
        return None

    required = {"tipo", "simbolo", value_col}
    if not required.issubset(by_symbol.columns):
        return None

    df = by_symbol.copy()
    df["tipo"] = df["tipo"].astype(str).replace({"": "Sin tipo"})
    df["simbolo"] = df["simbolo"].astype(str).replace({"": "Sin símbolo"})
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)

    pivot = (
        df.pivot_table(
            index="tipo",
            columns="simbolo",
            values=value_col,
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
    )

    if pivot.empty:
        return None

    tipos = pivot.index.tolist()
    symbols = pivot.columns.tolist()

    if pivot.values.size == 0:
        return None

    type_color_map = _color_discrete_map(pd.DataFrame({"tipo": tipos}))
    palette = list(dict.fromkeys(type_color_map.get(t, "#636EFA") for t in tipos))
    if len(palette) < 2:
        palette = palette * 2
    colorscale = [
        (i / (len(palette) - 1), color)
        for i, color in enumerate(palette)
    ]

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=pivot.values,
                x=symbols,
                y=tipos,
                colorscale=colorscale,
                colorbar=dict(title="% valorizado" if value_col.endswith("pct") else value_col),
                hovertemplate="Tipo: %{y}<br>Símbolo: %{x}<br>Valor: %{z:.2f}%<extra></extra>",
            )
        ]
    )

    if SHOW_AXIS_TITLES:
        fig.update_xaxes(title="Símbolo")
        fig.update_yaxes(title="Tipo")
    else:
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None)

    return _apply_layout(fig, show_legend=False)


# =================
# Gráficos avanzados
# =================

# def plot_bubble_pl_vs_costo(df: pd.DataFrame, x_axis: str, y_axis: str):
def plot_bubble_pl_vs_costo(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    *,
    color_seq: list[str] | None = None,
    log_x: bool = False,
    log_y: bool = False,
    category_col: str | None = None,
    benchmark_col: str | None = None,
):
    """Bubble chart flexible in axes, palette and scale."""
    if df is None or df.empty:
        return None
    needed = {x_axis, y_axis}
    if "simbolo" in df.columns:
        needed.add("simbolo")
    if not needed.issubset(df.columns):
        return None

    subset_cols = list({x_axis, y_axis, "valor_actual"})
    d = df.dropna(subset=[c for c in subset_cols if c in df.columns]).copy()
    if d.empty:
        return None

    # Evita tamaños negativos o NaN
    if "valor_actual" in d.columns:
        d["valor_ok"] = (
            pd.to_numeric(d["valor_actual"], errors="coerce")
            .clip(lower=0.0)
            .fillna(0.0)
        )
    else:
        d["valor_ok"] = 1.0

    color_kwargs: dict[str, Any] = {}
    if category_col and category_col in d.columns:
        pal = get_active_palette()
        categories = d[category_col].fillna("Desconocido").astype(str)
        unique_cats = categories.unique().tolist()
        color_map = {cat: pal.accent for cat in unique_cats}
        if "Benchmark" in color_map:
            color_map["Benchmark"] = pal.highlight_bg
        color_kwargs = {
            "color": category_col,
            "symbol": category_col,
            "color_discrete_map": None if color_seq else color_map,
            "color_discrete_sequence": color_seq,
        }
    elif "tipo" in d.columns:
        color_kwargs = {
            "color": "tipo",
            "color_discrete_map": None if color_seq else _color_discrete_map(d),
            "color_discrete_sequence": color_seq,
        }
    else:
        color_kwargs = {
            "color_discrete_sequence": color_seq or px.colors.qualitative.Plotly,
        }

    hover_data: dict[str, Any] = {"valor_ok": ":,.0f"}
    if "costo" in d:
        hover_data["costo"] = ":,.0f"
    if "pl" in d:
        hover_data["pl"] = ":,.0f"

    fig = px.scatter(
        d,
        x=x_axis,
        y=y_axis,
        size="valor_ok",
        hover_name="simbolo" if "simbolo" in d.columns else None,
        size_max=52,
        hover_data=hover_data,
        **color_kwargs,
    )
    # pal = get_active_palette()
    pal = get_active_palette()
    if SHOW_AXIS_TITLES:
        # fig.update_xaxes(title=x_axis.replace("_", " ").capitalize(), tickformat=",")
        # fig.update_yaxes(title=y_axis.replace("_", " ").capitalize(), tickformat=",", zeroline=True, zerolinecolor=pal.grid)
        fig.update_xaxes(
            title=x_axis.replace("_", " ").capitalize(),
            tickformat=",",
            type="log" if log_x else "linear",
        )
        fig.update_yaxes(
            title=y_axis.replace("_", " ").capitalize(),
            tickformat=",",
            zeroline=True,
            zerolinecolor=pal.grid,
            type="log" if log_y else "linear",
        )
    else:
        # fig.update_xaxes(title=None, tickformat=",")
        # fig.update_yaxes(title=None, tickformat=",", zeroline=True, zerolinecolor=pal.grid)
        fig.update_xaxes(tickformat=",", type="log" if log_x else "linear")
        fig.update_yaxes(tickformat=",", zeroline=True, zerolinecolor=pal.grid, type="log" if log_y else "linear")

    if benchmark_col and benchmark_col in d.columns:
        bench_points = d[d[benchmark_col].fillna(False)]
        if not bench_points.empty:
            bench_x = bench_points.iloc[0][x_axis]
            bench_y = bench_points.iloc[0][y_axis]
            if pd.notna(bench_x):
                fig.add_vline(
                    x=float(bench_x), line_dash="dot", line_color=pal.highlight_bg
                )
            if pd.notna(bench_y):
                fig.add_hline(
                    y=float(bench_y), line_dash="dot", line_color=pal.highlight_bg
                )

    return _apply_layout(fig)

#def plot_heat_pl_pct(df: pd.DataFrame):
def plot_heat_pl_pct(df: pd.DataFrame, color_scale: str = "RdBu"):
    if df is None or df.empty or "pl_%" not in df.columns or "simbolo" not in df.columns:
        return None
    d = df.dropna(subset=["pl_%"]).copy()
    if d.empty:
        return None
    # Escala divergente centrada en 0 para distinguir positivos/negativos
    vmax = float(d["pl_%"].abs().max())
    if vmax == 0:
        vmax = 1.0
    d = d.sort_values("pl_%", ascending=False)
    fig = px.bar(
        # d, x="simbolo", y="pl_%", color="pl_%",
        # color_continuous_scale="RdBu", range_color=[-vmax, vmax],
        d,
        x="simbolo",
        y="pl_%",
        color="pl_%",
        color_continuous_scale=color_scale,
        range_color=[-vmax, vmax],
        hover_data={"pl_%": ":.2f"},
        category_orders={"simbolo": d["simbolo"].astype(str).tolist()},
    )
    fig.update_traces(hovertemplate="%{x}: %{y:.2f}%<extra></extra>")
    pal = get_active_palette()
    if SHOW_AXIS_TITLES:
        fig.update_xaxes(title="símbolo")
        fig.update_yaxes(title="P/L %", zeroline=True, zerolinecolor=pal.grid)
    else:
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None, zeroline=True, zerolinecolor=pal.grid)
    return _apply_layout(fig, show_legend=False)

# ==============================
# P/L diaria (solo valores/delta)
# ==============================

def plot_pl_daily_topn(df: pd.DataFrame, n: int = 20):
    cols = [c for c in ("simbolo", "pl_d", "chg_%", "pld_%", "tipo") if c in df.columns]
    d = df[cols].copy() if cols else pd.DataFrame()

    if d.empty or "pl_d" not in d.columns:
        return None

    d["pl_d"] = pd.to_numeric(d["pl_d"], errors="coerce")
    if "chg_%" in d.columns:
        d["chg_%"] = pd.to_numeric(d["chg_%"], errors="coerce")
    if "pld_%" in d.columns:
        d["pld_%"] = pd.to_numeric(d["pld_%"], errors="coerce")
    if "chg_%" not in d.columns and "pld_%" in d.columns:
        d["chg_%"] = d["pld_%"]
    elif "pld_%" in d.columns:
        d["chg_%"] = d["chg_%"].fillna(d["pld_%"])

    d = d.dropna(subset=["pl_d"]).sort_values("pl_d", ascending=False).head(n)
    if d.empty:
        logger.warning("plot_pl_daily_topn: sin datos tras dropna")
        return None

    sym_map = _symbol_color_map(d["simbolo"].astype(str).tolist())

    custom_cols = ["pl_d"]
    if "chg_%" in d.columns:
        custom_cols.append("chg_%")

    fig = px.bar(
        d, x="simbolo", y="pl_d",
        color="simbolo",
        color_discrete_map=sym_map,
        hover_data={"tipo": True},
        custom_data=custom_cols,
    )

    if "chg_%" in d.columns:
        hover_tmpl = "<b>%{x}</b><br>P/L diaria: %{customdata[0]:,.0f}<br>Δ %: %{customdata[1]:.2f}%<extra></extra>"
    else:
        hover_tmpl = "<b>%{x}</b><br>P/L diaria: %{customdata[0]:,.0f}<extra></extra>"

    fig.update_traces(
        hovertemplate=hover_tmpl,
        texttemplate="%{y:,.0f}",
        textposition="outside",
        cliponaxis=False,
    )

    pal = get_active_palette()
    if SHOW_AXIS_TITLES:
        fig.update_xaxes(title="símbolo")
        fig.update_yaxes(title="P/L diario", tickformat=",", zeroline=True, zerolinecolor=pal.grid)
    else:
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None, tickformat=",", zeroline=True, zerolinecolor=pal.grid)

    return _apply_layout(fig, show_legend=False, y0_line=True)

# ============================
# (Opcional) comparativo VS
# ============================

def plot_pl_daily_vs_total(df: pd.DataFrame, n: int = 20):
    if df is None or df.empty:
        return None
    cols = ["simbolo","pl","pl_d","tipo"]
    if not set(["simbolo","pl"]).issubset(df.columns):
        return None
    d = df[[c for c in cols if c in df.columns]].copy()
    if "pl" not in d or d["pl"].dropna().empty:
        return None
    d = d.sort_values("pl", ascending=False).head(n)
    d_long = []
    for _, r in d.iterrows():
        d_long.append({"simbolo": r["simbolo"], "métrica": "P/L total",  "valor": r["pl"]})
        if pd.notna(r.get("pl_d")):
            d_long.append({"simbolo": r["simbolo"], "métrica": "P/L diaria", "valor": r["pl_d"]})
    d_long = pd.DataFrame(d_long)
    if d_long.empty:
        return None
    order = d["simbolo"].astype(str).tolist()
    pal = get_active_palette()
    fig = px.bar(
        d_long, x="simbolo", y="valor", color="métrica",
        barmode="group",
        color_discrete_sequence=[pal.accent, pal.positive],
        hover_data={"valor": ":,.0f"},
        category_orders={"simbolo": order},
    )
    fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside", cliponaxis=False)
    if SHOW_AXIS_TITLES:
        fig.update_xaxes(title="símbolo")
        fig.update_yaxes(title="Valor", tickformat=",", zeroline=True, zerolinecolor=pal.grid)
    else:
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None, tickformat=",", zeroline=True, zerolinecolor=pal.grid)
    return _apply_layout(fig, show_legend=True)

# ============================
# Subplots / Indicadores
# ============================

def plot_price_ma_bbands(df: pd.DataFrame, simbolo: str):
    """
    Precio con SMA/EMA + Bandas de Bollinger.
    Requiere columnas: Close, [SMA_FAST], [SMA_SLOW], [EMA], [BB_L, BB_M, BB_U] (opcionales)
    """
    if df is None or df.empty or "Close" not in df:
        return None
    fig = go.Figure()

    # Precio (línea)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name=f"{simbolo} Close", mode="lines", line=dict(width=2)
    ))

    # SMA/EMA (opcionales)
    if "SMA_FAST" in df: fig.add_trace(go.Scatter(x=df.index, y=df["SMA_FAST"], name="SMA corta", mode="lines"))
    if "SMA_SLOW" in df: fig.add_trace(go.Scatter(x=df.index, y=df["SMA_SLOW"], name="SMA larga", mode="lines"))
    if "EMA" in df:      fig.add_trace(go.Scatter(x=df.index, y=df["EMA"],      name="EMA", mode="lines"))

    # Bandas de Bollinger
    if {"BB_U","BB_L"}.issubset(df.columns):
        pal = get_active_palette()
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_U"], name="Banda Sup", mode="lines", line=dict(width=1)))
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_L"],
                name="Banda Inf",
                mode="lines",
                fill="tonexty",
                fillcolor=_rgba(pal.accent, 0.10),
                line=dict(width=1),
            )
        )
    if "BB_M" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_M"], name="Banda Media", mode="lines", line=dict(width=1, dash="dot")))

    pal = get_active_palette()
    fig.update_yaxes(tickformat=",", title=None, gridcolor=pal.grid)
    fig.update_xaxes(title=None, showgrid=False)
    return _apply_layout(fig, title=f"Precio + MAs + Bollinger — {simbolo}", show_legend=True)

def plot_rsi(df: pd.DataFrame):
    """RSI con líneas guía 70/30."""
    if df is None or df.empty or "RSI" not in df:
        return None
    fig = go.Figure()
    pal = get_active_palette()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", mode="lines"))
    fig.add_hrect(y0=70, y1=100, fillcolor=_rgba(pal.negative, 0.08), line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor=_rgba(pal.positive, 0.08), line_width=0)
    fig.update_yaxes(range=[0,100], title=None, gridcolor=pal.grid)
    fig.update_xaxes(title=None, showgrid=False)
    return _apply_layout(fig, title="RSI (14)", show_legend=False)

def plot_volume(df: pd.DataFrame):
    """Volumen en barras."""
    if df is None or df.empty or "Volume" not in df:
        return None
    fig = go.Figure()
    pal = get_active_palette()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volumen"))
    fig.update_yaxes(title=None, tickformat=",", gridcolor=pal.grid)
    fig.update_xaxes(title=None, showgrid=False)
    return _apply_layout(fig, title="Volumen", show_legend=False)

def plot_correlation_heatmap(prices_df: pd.DataFrame, *, title: str | None = None):
    """
    Calcula y grafica una matriz de correlación de los rendimientos diarios.
    """
    if prices_df is None or prices_df.empty or len(prices_df.columns) < 2:
        return None

    # 1) Rendimientos diarios (sin forward/back fill implícito)
    returns = prices_df.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(axis=0, how="all")
    if returns.empty:
        return None

    valid_variance = returns.var(skipna=True)
    if valid_variance.empty:
        return None

    valid_columns = valid_variance[valid_variance > 0].index.tolist()
    if len(valid_columns) < 2:
        return None

    returns = returns[valid_columns]
    returns = returns.dropna(axis=1, how="all")
    if returns.empty or len(returns.columns) < 2:
        return None

    # 2) Matriz de correlación
    corr_matrix = returns.corr()
    if corr_matrix.empty or len(corr_matrix.columns) < 2:
        return None

    # 3) Heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        range_color=[-1, 1],
    )
    fig.update_traces(hovertemplate='Correlación entre %{x} y %{y}:<br><b>%{z:.2f}</b><extra></extra>')
    fig.update_xaxes(side="bottom")
    chart_title = title or "Matriz de Correlación de Rendimientos Diarios"
    return _apply_layout(fig, title=chart_title)

def plot_technical_analysis_chart(df_ind: pd.DataFrame, sma_fast: int, sma_slow: int):
    """
    Gráfico compuesto con precio e indicadores técnicos.
    Requiere columna Close; opcionales SMA/EMA/Bollinger/RSI/MACD/Stoch/ATR/Ichimoku/Volume
    """
    if df_ind is None or df_ind.empty or "Close" not in df_ind:
        return None

    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.35, 0.15, 0.15, 0.15, 0.10, 0.10],
    )

    # 1) Precio + Indicadores
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Close"], name="Close", mode="lines"), row=1, col=1)
    if "SMA_FAST" in df_ind:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["SMA_FAST"], name=f"SMA {sma_fast}", mode="lines"), row=1, col=1)
    if "SMA_SLOW" in df_ind:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["SMA_SLOW"], name=f"SMA {sma_slow}", mode="lines"), row=1, col=1)
    if "EMA" in df_ind:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["EMA"], name="EMA", mode="lines"), row=1, col=1)
    if "BB_U" in df_ind:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_U"], name="BB Upper", line=dict(dash="dot")), row=1, col=1)
    if "BB_L" in df_ind:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_L"], name="BB Lower", line=dict(dash="dot")), row=1, col=1)
    if {"ICHI_CONV", "ICHI_BASE"}.issubset(df_ind.columns):
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["ICHI_CONV"], name="Ichimoku Conv", mode="lines"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["ICHI_BASE"], name="Ichimoku Base", mode="lines"), row=1, col=1)
    if {"ICHI_A", "ICHI_B"}.issubset(df_ind.columns):
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["ICHI_A"], name="Ichimoku A", line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["ICHI_B"], name="Ichimoku B", fill="tonexty", line=dict(width=0), fillcolor="rgba(200,200,200,0.2)", showlegend=False), row=1, col=1)
    fig.update_yaxes(title_text="Precio", row=1, col=1)

    # 2) MACD (si existe)
    if "MACD" in df_ind:
        fig.add_trace(go.Bar(x=df_ind.index, y=df_ind.get("MACD_HIST"), name="MACD Hist"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD"], name="MACD", mode="lines"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD_SIGNAL"], name="Señal", mode="lines"), row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)

    # 3) RSI
    if "RSI" in df_ind:
        # fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["RSI"], name="RSI", mode="lines"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["RSI"], name="RSI", mode="lines"), row=3, col=1)
        pal = get_active_palette()

    # # 3) Volumen (si existe)
        fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor=_rgba(pal.negative, 0.08), row=3, col=1)
        fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor=_rgba(pal.positive, 0.08), row=3, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

    # 4) Estocástico
    if "STOCH_K" in df_ind:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["STOCH_K"], name="%K", mode="lines"), row=4, col=1)
        if "STOCH_D" in df_ind:
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["STOCH_D"], name="%D", mode="lines"), row=4, col=1)
        fig.update_yaxes(title_text="Estocástico", range=[0, 100], row=4, col=1)

    # 5) ATR
    if "ATR" in df_ind:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["ATR"], name="ATR", mode="lines"), row=5, col=1)
        fig.update_yaxes(title_text="ATR", row=5, col=1)

    # 6) Volumen
    if "Volume" in df_ind:
        fig.add_trace(go.Bar(x=df_ind.index, y=df_ind["Volume"], name="Volumen"), row=6, col=1)
        fig.update_yaxes(title_text="Volumen", row=6, col=1)
        
    _apply_layout(fig, show_legend=True)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
    return fig
