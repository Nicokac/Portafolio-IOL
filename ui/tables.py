# tables.py
from __future__ import annotations
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

    cols_order = ["mercado","simbolo","tipo","cantidad","ultimo","valor_actual","costo","pl","pl_%","pl_d","chg_%"]
    for c in cols_order:
        if c not in df_view.columns:
            df_view[c] = np.nan

    try:
        df_sorted = df_view.sort_values(order_by, ascending=not desc).copy()
    except Exception:
        df_sorted = df_view.copy()

    quotes_hist: dict = st.session_state.get("quotes_hist", {})
    SPARK_N = 30

    fmt_rows = []
    all_spark_vals: list[float] = []

    for _, r in df_sorted.iterrows():
        sym = str(r["simbolo"])
        tipo = str(r.get("tipo") or "")
        cur = _detect_currency(sym, tipo)

        row = {
            "BCBA": str(r["mercado"]),
            "Símbolo": sym,
            "Tipo": tipo,
            "Cantidad": format_number(r["cantidad"]),
            "Último precio": format_price(r["ultimo"], currency=cur),
            "Valorizado": format_money(r["valor_actual"], currency=cur),
            "Costo": format_money(r["costo"], currency=cur),
        }

        pl_val = r.get("pl")
        pl_pct_val = r.get("pl_%")
        if not _is_none_nan_inf(pl_val):
            label_pl = format_money(pl_val, currency=cur)
            if not _is_none_nan_inf(pl_pct_val):
                label_pl += f" ({float(pl_pct_val):.2f}%)"
        else:
            label_pl = "—"
        row["P/L Acumulado"] = label_pl

        pl_d_val = r.get("pl_d")
        chg_pct = r.get("chg_%")
        if not _is_none_nan_inf(pl_d_val):
            label_pld = format_money(pl_d_val, currency=cur)
            if not _is_none_nan_inf(chg_pct):
                label_pld += f" ({float(chg_pct):.2f} %)"
        else:
            label_pld = "—"
        row["P/L diaria"] = label_pld

        hist = quotes_hist.get(sym.upper(), [])
        vals = []
        for h in hist:
            v = _as_float_or_none(h.get("chg_pct"))
            if v is not None:
                vals.append(v)
        vals = vals[-SPARK_N:]
        row["Intradía %"] = vals if len(vals) >= 2 else None
        all_spark_vals.extend(vals)

        if show_usd and _as_float_or_none(ccl_rate):
            rate = float(ccl_rate)
            row["Val. (USD CCL)"]  = format_money((float(r["valor_actual"])/rate) if not _is_none_nan_inf(r["valor_actual"]) else None, "USD")
            row["Costo (USD CCL)"] = format_money((float(r["costo"])/rate) if not _is_none_nan_inf(r["costo"]) else None, "USD")
            row["P/L (USD CCL)"]   = format_money((float(r["pl"])/rate) if not _is_none_nan_inf(r["pl"]) else None, "USD")

        fmt_rows.append(row)

    df_tbl = pd.DataFrame(fmt_rows)

    def _color_pl(col: pd.Series):
        styles = []
        for v in col:
            s = str(v or "").strip()
            if s.startswith("-"):
                styles.append("color: #E57373; font-weight: 600;")
            elif s not in {"—", ""}:
                styles.append("color: #81C784; font-weight: 600;")
            else:
                styles.append("")
        return styles

    if all_spark_vals:
        span = max(abs(min(all_spark_vals)), abs(max(all_spark_vals)))
        span = max(2.0, min(span, 30.0))
        y_min, y_max = -span, span
    else:
        y_min, y_max = -10.0, 10.0

    st.subheader("Detalle por símbolo")
    st.dataframe(
        df_tbl.style.apply(_color_pl, subset=["P/L Acumulado", "P/L diaria"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Intradía %": st.column_config.LineChartColumn(
                label="Intradía %",
                width="small",
                y_min=y_min,
                y_max=y_max,
                help="Variación diaria (%) intradía — últimos puntos",
            )
        },
    )
