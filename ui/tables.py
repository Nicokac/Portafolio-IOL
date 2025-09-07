from __future__ import annotations
import math
import numpy as np
import pandas as pd
import streamlit as st
from shared.utils import (
    _as_float_or_none,
    _is_none_nan_inf,
    format_money,
    format_number,
    format_price,
)
from .palette import get_active_palette
from .export import download_csv

def render_totals(df_view: pd.DataFrame, ccl_rate: float | None = None):
    total_val  = float(np.nansum(df_view.get("valor_actual", pd.Series(dtype=float)).values))
    total_cost = float(np.nansum(df_view.get("costo", pd.Series(dtype=float)).values))
    total_pl   = total_val - total_cost
    total_pl_pct = (total_pl / total_cost * 100.0) if total_cost else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valorizado", format_money(total_val))
    c2.metric("Costo", format_money(total_cost))
    c3.metric("P/L", format_money(total_pl), delta=None if not np.isfinite(total_pl_pct) else f"{total_pl_pct:.2f}%")
    c4.metric("P/L %", "—" if not np.isfinite(total_pl_pct) else f"{total_pl_pct:.2f}%")

    if _as_float_or_none(ccl_rate):
        rate = float(ccl_rate)
        usd_val  = total_val / rate
        usd_cost = total_cost / rate
        usd_pl   = total_pl / rate
        c1b, c2b, c3b, c4b = st.columns(4)
        c1b.metric("Valorizado (USD CCL)", format_money(usd_val, currency="USD"))
        c2b.metric("Costo (USD CCL)", format_money(usd_cost, currency="USD"))
        c3b.metric("P/L (USD CCL)", format_money(usd_pl, currency="USD"))
        c4b.metric("CCL usado", format_money(rate))

def _detect_currency(sym: str, tipo: str | None) -> str:
    return "USD" if str(sym).upper() in {"PRPEDOB"} else "ARS"

def render_table(df_view: pd.DataFrame, order_by: str, desc: bool, ccl_rate: float | None = None, show_usd: bool = False):
    if df_view is None or df_view.empty:
        st.info("Sin datos para mostrar.")
        return

    palette = get_active_palette()
    st.markdown(f"""
        <style>
        div[data-testid='stDataFrame'] thead tr th {{
            position: sticky;
            top: 0;
            background-color: {palette.bg};
            color: {palette.text};
        }}
        div[data-testid='stDataFrame'] tbody tr {{
            background-color: {palette.bg};
            color: {palette.text};
        }}
        div[data-testid='stDataFrame'] tbody td {{
            transition: background-color 0.2s ease;
        }}
        div[data-testid='stDataFrame'] tbody td:hover {{
            background-color: {palette.highlight_bg};
            color: {palette.highlight_text};
        }}
        </style>
    """, unsafe_allow_html=True)

    cols_order = ["mercado", "simbolo", "tipo", "cantidad", "ultimo", "valor_actual", "costo", "pl", "pl_%", "pl_d", "chg_%"]
    for c in cols_order:
        if c not in df_view.columns:
            df_view[c] = np.nan

    try:
        df_sorted = df_view.sort_values(order_by, ascending=not desc).copy()
    except Exception:
        df_sorted = df_view.copy()

    search = st.text_input("Buscar", "").strip().lower()
    if search:
        mask = (
            df_sorted["simbolo"].astype(str).str.lower().str.contains(search)
            | df_sorted["tipo"].astype(str).str.lower().str.contains(search)
        )
        df_sorted = df_sorted[mask]

    if df_sorted.empty:
        st.info("Sin datos para mostrar.")
        return

    quotes_hist: dict = st.session_state.get("quotes_hist", {})
    SPARK_N = 30

    fmt_rows = []
    chg_list: list[float | None] = []
    all_spark_vals: list[float] = []

    for _, r in df_sorted.iterrows():
        sym = str(r["simbolo"])
        tipo = str(r.get("tipo") or "")
        cur = _detect_currency(sym, tipo)

        row = {
            "Símbolo": sym,
            "Tipo": tipo,
            "Cantidad": format_number(r["cantidad"]),
            "Último precio": format_price(r["ultimo"], currency=cur),
            "Valorizado": format_money(r["valor_actual"], currency=cur),
            "Costo": format_money(r["costo"], currency=cur),
        }

        pl_val = r.get("pl")
        pl_pct_val = r.get("pl_%")
        row["P/L Acum Valor"] = format_money(pl_val, currency=cur) if not _is_none_nan_inf(pl_val) else "—"
        row["P/L Acum %"] = f"{float(pl_pct_val):.2f}%" if not _is_none_nan_inf(pl_pct_val) else "—"

        pl_d_val = r.get("pl_d")
        chg_pct = r.get("chg_%")
        row["P/L diario Valor"] = format_money(pl_d_val, currency=cur) if not _is_none_nan_inf(pl_d_val) else "—"
        row["P/L diario %"] = f"{float(chg_pct):.2f}%" if not _is_none_nan_inf(chg_pct) else "—"
        chg_list.append(_as_float_or_none(chg_pct))

        hist = quotes_hist.get(sym.upper(), [])
        vals = [
            _as_float_or_none(h.get("chg_pct"))
            for h in hist[-SPARK_N:] if _as_float_or_none(h.get("chg_pct")) is not None
        ]
        row["Intradía %"] = vals if len(vals) >= 2 else None
        all_spark_vals.extend(vals)

        if show_usd and _as_float_or_none(ccl_rate):
            rate = float(ccl_rate)
            row["Val. (USD CCL)"] = format_money((float(r["valor_actual"]) / rate) if not _is_none_nan_inf(r["valor_actual"]) else None, "USD")
            row["Costo (USD CCL)"] = format_money((float(r["costo"]) / rate) if not _is_none_nan_inf(r["costo"]) else None, "USD")
            row["P/L (USD CCL)"] = format_money((float(r["pl"]) / rate) if not _is_none_nan_inf(r["pl"]) else None, "USD")

        fmt_rows.append(row)

    df_tbl = pd.DataFrame(fmt_rows)

    def _color_pl(col: pd.Series):
        styles = []
        for v in col:
            s = str(v or "").strip()
            if s.startswith("-"):
                styles.append(f"color: {palette.negative}; font-weight: 600;")
            elif s not in {"—", ""}:
                styles.append(f"color: {palette.positive}; font-weight: 600;")
            else:
                styles.append("")
        return styles

    if all_spark_vals:
        span = max(abs(min(all_spark_vals)), abs(max(all_spark_vals)))
        span = max(2.0, min(span, 30.0))
        y_min, y_max = -span, span
    else:
        y_min, y_max = -10.0, 10.0

    column_help = {
        "Símbolo": "Ticker del activo",
        "Tipo": "Clasificación del instrumento",
        "Cantidad": "Cantidad de títulos en cartera",
        "Último precio": "Última cotización disponible",
        "Valorizado": "Valor actual (cantidad * último precio)",
        "Costo": "Costo total de adquisición",
        "P/L Acum Valor": "Ganancia/Pérdida desde la compra",
        "P/L Acum %": "Ganancia/Pérdida porcentual desde la compra",
        "P/L diario Valor": "Ganancia/Pérdida de la rueda actual",
        "P/L diario %": "Variación porcentual de la rueda actual",
        "Val. (USD CCL)": "Valorizado en USD usando CCL",
        "Costo (USD CCL)": "Costo en USD usando CCL",
        "P/L (USD CCL)": "Ganancia/Pérdida en USD usando CCL",
    }

    column_config: dict[str, st.column_config.Column] = {}
    for col, help_txt in column_help.items():
        if col in df_tbl.columns:
            column_config[col] = st.column_config.Column(label=col, help=help_txt)

    column_config["Intradía %"] = st.column_config.LineChartColumn(
        label="Intradía %",
        width="small",
        y_min=y_min,
        y_max=y_max,
        help="Variación diaria (%) intradía — últimos puntos",
    )

    st.subheader("Detalle por símbolo")
    download_csv(df_tbl, "portafolio.csv")

    page_size = st.number_input("Filas por página", min_value=5, max_value=100, value=20, step=5)
    total_pages = max(1, math.ceil(len(df_tbl) / page_size))
    page = st.number_input("Página", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = start + page_size

    df_page = df_tbl.iloc[start:end]

    st.dataframe(
        df_page.style
            .apply(_color_pl, subset=["P/L Acum Valor", "P/L diario Valor", "P/L Acum %", "P/L diario %"]),
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config=column_config,
    )
