# ui\fx_panels.py
from __future__ import annotations
import html
import pandas as pd
import streamlit as st
import plotly.express as px
from shared.utils import _as_float_or_none, format_percent
from .palette import get_active_palette

# def render_fx_panel(rates: dict):
#     st.subheader("💵 Cotizaciones del dólar (ARS por USD)")
#     if not rates:
#         st.info("No se pudieron obtener cotizaciones en este momento.")
#         return

#     rows = []
#     order = ["oficial", "mayorista", "ahorro", "tarjeta", "blue", "mep", "ccl", "cripto"]
#     labels = {
#         "oficial": "Oficial",
#         "mayorista": "Mayorista",
#         "ahorro": "Ahorro / Solidario",
#         "tarjeta": "Tarjeta / Turista",
#         "blue": "Blue",
#         "mep": "MEP (Bolsa)",
#         "ccl": "CCL (Contado c/Liq.)",
#         "cripto": "Cripto",
#     }
#     descriptions = {
#     "oficial": "Tipo de cambio oficial minorista",
#     "mayorista": "Precio mayorista de referencia",
#     "ahorro": "Oficial + impuestos",
#     "tarjeta": "Consumos con tarjeta en el exterior",
#     "blue": "Mercado paralelo",
#     "mep": "Compra/venta de bonos en pesos",
#     "ccl": "Contado con liquidación, referencia para CEDEARs/ADRs",
#     "cripto": "Cotización en exchanges cripto",
#     }  

#     for k in order:
#         if k in rates and _as_float_or_none(rates[k]) is not None:
#             val = float(rates[k])
#             label = labels.get(k, k)
#             rows.append(
#                 {
#                     "Tipo": label,
#                     "ARS / USD": f"$ {val:,.2f}".replace(",", "_").replace(".", ",").replace("_", "."),
#                     "Ref": "CCL" if k == "ccl" else "",
#                     "Desc": descriptions.get(k, ""),
#                 }
#             )

#     if not rows:
#         st.info("No hay datos de tipos de cambio para mostrar.")
#         return

#     df_fx = pd.DataFrame(rows)

#     pal = get_active_palette()
#     st.markdown(
#         f"""
#         <style>
#         .fx-badge {{
#             background-color: {pal.highlight_bg};
#             color: {pal.highlight_text};
#             padding: 0 0.4em;
#             border-radius: 0.25rem;
#             font-size: 0.75em;
#             font-weight: 600;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

#     df_fx["Ref"] = df_fx["Ref"].apply(lambda v: f"<span class='fx-badge'>{v}</span>" if v else "")
#     df_fx["Tipo"] = df_fx.apply(
#         lambda r: f"<span title='{html.escape(r['Desc'])}'>{r['Tipo']}</span>", axis=1
#     )
#     df_fx = df_fx.drop(columns=["Desc"])

#     st.write(df_fx.to_html(escape=False, index=False), unsafe_allow_html=True)
#     st.caption("Nota: el **CCL** es la referencia usual para CEDEARs/ADRs/bonos en USD.")

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
    df_long = hist[["ts_dt"] + cols].melt("ts_dt", var_name="Tipo", value_name="ARS")
    fig = px.line(df_long, x="ts_dt", y="ARS", color="Tipo", hover_name="Tipo")
    fig.update_layout(xaxis_title="", yaxis_title="ARS / USD", legend_title_text="Tipo")
    st.plotly_chart(fig, width="stretch")
    st.caption("Línea que refleja cómo cambian las cotizaciones del dólar a lo largo del día.")

