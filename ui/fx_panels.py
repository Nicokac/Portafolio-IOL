# ui\fx_panels.py
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from shared.utils import _as_float_or_none, format_percent


def render_spreads(rates: dict):
    st.subheader("🔀 Brechas de dólar")
    if not rates:
        st.info("Sin cotizaciones para calcular brechas.")
        return

    ccl = _as_float_or_none(rates.get("ccl"))
    oficial = _as_float_or_none(rates.get("oficial"))
    blue = _as_float_or_none(rates.get("blue"))
    mep = _as_float_or_none(rates.get("mep"))
    mayorista = _as_float_or_none(rates.get("mayorista"))

    def pct(a, b):
        if a is None or b is None or a <= 0 or b <= 0:
            return None
        return (a / b - 1.0) * 100.0

    rows = [
        {"Par": "Oficial vs CCL", "Brecha": format_percent(pct(ccl, oficial))},
        {"Par": "Blue vs CCL", "Brecha": format_percent(pct(ccl, blue))},
        {"Par": "MEP vs CCL", "Brecha": format_percent(pct(ccl, mep))},
    ]
    if mayorista:
        rows.append({"Par": "Mayorista vs CCL", "Brecha": format_percent(pct(ccl, mayorista))})
    rows.append({"Par": "Blue vs Oficial", "Brecha": format_percent(pct(blue, oficial))})

    _ = st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Muestra la diferencia porcentual entre distintas cotizaciones del dólar.")


def render_fx_history(history: pd.DataFrame):
    st.subheader("⏱️ Serie intradía del dólar")
    if history is None or history.empty:
        st.info("Aún no hay historial para graficar.")
        return
    hist = history.sort_values("ts_dt")
    cols = [c for c in ["ccl", "mep", "blue", "oficial"] if c in hist.columns]
    if not cols:
        st.info("No hay series disponibles para graficar.")
        return
    _ = st.line_chart(hist[cols])
    st.caption("Evolución intradía de las cotizaciones seleccionadas")
    df_long = hist[["ts_dt"] + cols].melt("ts_dt", var_name="Tipo", value_name="ARS")
    fig = px.line(df_long, x="ts_dt", y="ARS", color="Tipo", hover_name="Tipo")
    fig.update_layout(xaxis_title="", yaxis_title="ARS / USD", legend_title_text="Tipo")
    st.plotly_chart(fig, config={"responsive": True})
    st.caption("Línea que refleja cómo cambian las cotizaciones del dólar a lo largo del día.")
