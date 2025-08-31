# ui/charts.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# Tema / colores unificados
# =========================
BG          = "#0e1117"
PLOT_BG     = "#0e1117"
GRID_COLOR  = "rgba(255,255,255,0.08)"
FONT_COLOR  = "#e5e5e5"
FONT_FAMILY = "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"

# Mostrar/ocultar títulos de ejes desde un flag único
SHOW_AXIS_TITLES = True

# Paleta clara por tipo
PALETTE = {
    "CEDEAR": "#7DB3FF",
    "Bono":   "#8EE0A1",
    "Acción": "#F7A6A6",
    "ETF":    "#C9B6FF",
    "FCI":    "#A7E3EB",
    "Letra":  "#B6E08E",
    "Otro":   "#E0E0E0",
}

# Paleta estable por símbolo
_SYMBOL_PALETTE = (
    px.colors.qualitative.Set2
    + px.colors.qualitative.Pastel
    + px.colors.qualitative.Light24
)

def _symbol_color_map(symbols: list[str]) -> dict[str, str]:
    return {s: _SYMBOL_PALETTE[i % len(_SYMBOL_PALETTE)] for i, s in enumerate(symbols)}

def _color_discrete_map(df: pd.DataFrame, tipo_col: str = "tipo"):
    tipos = [t for t in df[tipo_col].dropna().unique().tolist()]
    return {t: PALETTE.get(t, "#D8D8D8") for t in tipos}

def _si(n: float) -> str:
    try:
        return f"{n:,.0f}".replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception:
        return str(n)

def _apply_layout(fig: go.Figure, title: str | None = None, *, show_legend=True, y0_line=False):
    fig.update_layout(
        title=({"text": title, "x": 0.0, "xanchor": "left"} if title else None),
        font=dict(family=FONT_FAMILY, color=FONT_COLOR, size=14),
        paper_bgcolor=BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=16, r=12, t=48 if title else 24, b=32),
        legend_title_text="",
        showlegend=show_legend,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=y0_line, zerolinecolor=GRID_COLOR)
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

# =================
# Gráficos avanzados
# =================

def plot_bubble_pl_vs_costo(df: pd.DataFrame, x_axis: str, y_axis: str):
    if df is None or df.empty:
        return None
    needed = {"simbolo", "tipo", "valor_actual", x_axis, y_axis}
    if not needed.issubset(df.columns):
        return None

    subset_cols = list(needed | {"costo", "pl"})
    d = df.dropna(subset=[c for c in subset_cols if c in df.columns]).copy()
    if d.empty:
        return None

    # Evita tamaños negativos o NaN
    d["valor_ok"] = pd.to_numeric(d["valor_actual"], errors="coerce").clip(lower=0.0).fillna(0.0)

    fig = px.scatter(
        d, x=x_axis, y=y_axis, size="valor_ok", color="tipo",
        hover_name="simbolo", size_max=52,
        color_discrete_map=_color_discrete_map(d),
        hover_data={"costo": ":,.0f" if "costo" in d else True,
                    "pl": ":,.0f" if "pl" in d else True,
                    "valor_ok": ":,.0f"},
    )
    if SHOW_AXIS_TITLES:
        fig.update_xaxes(title=x_axis.replace("_", " ").capitalize(), tickformat=",")
        fig.update_yaxes(title=y_axis.replace("_", " ").capitalize(), tickformat=",", zeroline=True, zerolinecolor=GRID_COLOR)
    else:
        fig.update_xaxes(title=None, tickformat=",")
        fig.update_yaxes(title=None, tickformat=",", zeroline=True, zerolinecolor=GRID_COLOR)
    return _apply_layout(fig)

def plot_heat_pl_pct(df: pd.DataFrame):
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
        d, x="simbolo", y="pl_%", color="pl_%",
        color_continuous_scale="RdBu", range_color=[-vmax, vmax],
        hover_data={"pl_%": ":.2f"},
        category_orders={"simbolo": d["simbolo"].astype(str).tolist()},
    )
    fig.update_traces(hovertemplate="%{x}: %{y:.2f}%<extra></extra>")
    if SHOW_AXIS_TITLES:
        fig.update_xaxes(title="símbolo")
        fig.update_yaxes(title="P/L %", zeroline=True, zerolinecolor=GRID_COLOR)
    else:
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None, zeroline=True, zerolinecolor=GRID_COLOR)
    return _apply_layout(fig, show_legend=False)

# ==============================
# P/L diaria (solo valores/delta)
# ==============================

def plot_pl_daily_topn(df: pd.DataFrame, n: int = 20):
    cols = [c for c in ("simbolo","pl_d","chg_%","tipo") if c in df.columns]
    d = df[cols].copy() if cols else pd.DataFrame()

    if d.empty or "pl_d" not in d.columns:
        return None

    d["pl_d"] = pd.to_numeric(d["pl_d"], errors="coerce")
    if "chg_%" in d.columns:
        d["chg_%"] = pd.to_numeric(d["chg_%"], errors="coerce")

    d = d.dropna(subset=["pl_d"]).sort_values("pl_d", ascending=False).head(n)
    if d.empty:
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

    if SHOW_AXIS_TITLES:
        fig.update_xaxes(title="símbolo")
        fig.update_yaxes(title="P/L diario", tickformat=",", zeroline=True, zerolinecolor=GRID_COLOR)
    else:
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None, tickformat=",", zeroline=True, zerolinecolor=GRID_COLOR)

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
    fig = px.bar(
        d_long, x="simbolo", y="valor", color="métrica",
        barmode="group",
        color_discrete_sequence=["#6EA8FE","#9AD0F5"],
        hover_data={"valor": ":,.0f"},
        category_orders={"simbolo": order},
    )
    fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside", cliponaxis=False)
    if SHOW_AXIS_TITLES:
        fig.update_xaxes(title="símbolo")
        fig.update_yaxes(title="Valor", tickformat=",", zeroline=True, zerolinecolor=GRID_COLOR)
    else:
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None, tickformat=",", zeroline=True, zerolinecolor=GRID_COLOR)
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
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_U"], name="Banda Sup", mode="lines", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_L"], name="Banda Inf", mode="lines", fill="tonexty", fillcolor="rgba(125,179,255,0.10)", line=dict(width=1)))
    if "BB_M" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_M"], name="Banda Media", mode="lines", line=dict(width=1, dash="dot")))

    fig.update_yaxes(tickformat=",", title=None, gridcolor=GRID_COLOR)
    fig.update_xaxes(title=None, showgrid=False)
    return _apply_layout(fig, title=f"Precio + MAs + Bollinger — {simbolo}", show_legend=True)

def plot_rsi(df: pd.DataFrame):
    """RSI con líneas guía 70/30."""
    if df is None or df.empty or "RSI" not in df:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", mode="lines"))
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,0,0,0.08)", line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(0,255,0,0.08)", line_width=0)
    fig.update_yaxes(range=[0,100], title=None, gridcolor=GRID_COLOR)
    fig.update_xaxes(title=None, showgrid=False)
    return _apply_layout(fig, title="RSI (14)", show_legend=False)

def plot_volume(df: pd.DataFrame):
    """Volumen en barras."""
    if df is None or df.empty or "Volume" not in df:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volumen"))
    fig.update_yaxes(title=None, tickformat=",", gridcolor=GRID_COLOR)
    fig.update_xaxes(title=None, showgrid=False)
    return _apply_layout(fig, title="Volumen", show_legend=False)

def plot_correlation_heatmap(prices_df: pd.DataFrame):
    """
    Calcula y grafica una matriz de correlación de los rendimientos diarios.
    """
    if prices_df is None or prices_df.empty or len(prices_df.columns) < 2:
        return None

    # 1) Rendimientos diarios (sin forward/back fill implícito)
    returns = prices_df.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
    if returns.empty or len(returns.columns) < 2:
        return None

    # 2) Matriz de correlación
    corr_matrix = returns.corr()

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
    return _apply_layout(fig, title="Matriz de Correlación de Rendimientos Diarios")

def plot_technical_analysis_chart(df_ind: pd.DataFrame, sma_fast: int, sma_slow: int):
    """
    Crea el gráfico compuesto de análisis técnico con subplots.
    Requiere columnas: Close; opcionales: SMA_FAST, SMA_SLOW, EMA, BB_L/BB_M/BB_U, RSI, Volume
    """
    if df_ind is None or df_ind.empty or "Close" not in df_ind:
        return None

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.20, 0.25],
    )

    # 1) Precio + Indicadores
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Close"], name="Close", mode="lines"), row=1, col=1)
    if "SMA_FAST" in df_ind: fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["SMA_FAST"], name=f"SMA {sma_fast}", mode="lines"), row=1, col=1)
    if "SMA_SLOW" in df_ind: fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["SMA_SLOW"], name=f"SMA {sma_slow}", mode="lines"), row=1, col=1)
    if "EMA" in df_ind:      fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["EMA"], name="EMA 21", mode="lines"), row=1, col=1)
    if "BB_U" in df_ind:     fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_U"], name="BB Upper", line=dict(dash="dot")), row=1, col=1)
    if "BB_L" in df_ind:     fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_L"], name="BB Lower", line=dict(dash="dot")), row=1, col=1)
    fig.update_yaxes(title_text="Precio", row=1, col=1)

    # 2) RSI (si existe)
    if "RSI" in df_ind:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["RSI"], name="RSI", mode="lines"), row=2, col=1)
        fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.08, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.08, row=2, col=1)
        fig.update_yaxes(title_text="RSI (14)", range=[0, 100], row=2, col=1)

    # 3) Volumen (si existe)
    if "Volume" in df_ind:
        fig.add_trace(go.Bar(x=df_ind.index, y=df_ind["Volume"], name="Volumen"), row=3, col=1)
        fig.update_yaxes(title_text="Volumen", row=3, col=1)

    _apply_layout(fig, show_legend=True)
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
    return fig
