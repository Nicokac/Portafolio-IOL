# ui\fx_panels.py
from __future__ import annotations
import pandas as pd
import streamlit as st
from shared.utils import _as_float_or_none, format_percent

def _highlight_ccl(row):
    if str(row.get("Tipo", "")).upper() == "CCL":
        return ["background-color: #1b5e20; color: white; font-weight: 700"] * len(row)
    return [""] * len(row)

def render_fx_panel(rates: dict):
    st.subheader("💵 Cotizaciones del dólar (ARS por USD)")
    if not rates:
        st.info("No se pudieron obtener cotizaciones en este momento.")
        return

    rows = []
    order = ["oficial","mayorista","ahorro","tarjeta","blue","mep","ccl","cripto"]
    labels = {
        "oficial": "Oficial",
        "mayorista": "Mayorista",
        "ahorro": "Ahorro / Solidario",
        "tarjeta": "Tarjeta / Turista",
        "blue": "Blue",
        "mep": "MEP (Bolsa)",
        "ccl": "CCL (Contado c/Liq.)",
        "cripto": "Cripto",
    }

    for k in order:
        if k in rates and _as_float_or_none(rates[k]) is not None:
            val = float(rates[k])
            rows.append({
                "Tipo": "CCL" if k == "ccl" else labels.get(k, k),
                "ARS / USD": f"$ {val:,.2f}".replace(",", "_").replace(".", ",").replace("_",".")
            })

    if not rows:
        st.info("No hay datos de tipos de cambio para mostrar.")
        return

    df_fx = pd.DataFrame(rows)
    _ = st.dataframe(df_fx.style.apply(_highlight_ccl, axis=1), use_container_width=True, hide_index=True)
    st.caption("Nota: el **CCL** es la referencia usual para CEDEARs/ADRs/bonos en USD.")

def render_spreads(rates: dict):
    st.subheader("🔀 Brechas de dólar")
    if not rates:
        st.info("Sin cotizaciones para calcular brechas.")
        return

    ccl       = _as_float_or_none(rates.get("ccl"))
    oficial   = _as_float_or_none(rates.get("oficial"))
    blue      = _as_float_or_none(rates.get("blue"))
    mep       = _as_float_or_none(rates.get("mep"))
    mayorista = _as_float_or_none(rates.get("mayorista"))

    def pct(a, b):
        if a is None or b is None or a <= 0 or b <= 0:
            return None
        return (a / b - 1.0) * 100.0

    rows = [
        {"Par": "Oficial vs CCL",   "Brecha": format_percent(pct(ccl, oficial))},
        {"Par": "Blue vs CCL",      "Brecha": format_percent(pct(ccl, blue))},
        {"Par": "MEP vs CCL",       "Brecha": format_percent(pct(ccl, mep))},
    ]
    if mayorista:
        rows.append({"Par": "Mayorista vs CCL", "Brecha": format_percent(pct(ccl, mayorista))})
    rows.append({"Par": "Blue vs Oficial", "Brecha": format_percent(pct(blue, oficial))})

    _ = st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render_fx_history(history: pd.DataFrame):
    st.subheader("⏱️ Serie intradía del dólar")
    if history is None or history.empty:
        st.info("Aún no hay historial para graficar.")
        return
    hist = history.sort_values("ts_dt").set_index("ts_dt")
    cols = [c for c in ["ccl","mep","blue","oficial"] if c in hist.columns]
    if not cols:
        st.info("No hay series disponibles para graficar.")
        return
    _ = st.line_chart(hist[cols])
