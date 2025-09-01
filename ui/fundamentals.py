# ui\fundamentals.py
from __future__ import annotations
import streamlit as st
from shared.utils import _is_none_nan_inf, format_number, format_percent

def render_fundamental_data(data: dict):
    if not data or data.get("error"):
        st.warning(data.get("error", "Datos fundamentales no disponibles."))
        return

    st.subheader(f"Análisis Fundamental: {data.get('name', '—')}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Capitalización de Mercado", f"US$ {format_number(data.get('market_cap'))}")

    pe = data.get("pe_ratio")
    col2.metric("Ratio P/E (TTM)", "—" if _is_none_nan_inf(pe) else f"{float(pe):.2f}")

    col3.metric("Rendimiento por Dividendo", format_percent(data.get("dividend_yield")))

    st.caption(f"**Sector:** {data.get('sector', '—')} | **Web:** {data.get('website', '—')}")
    st.divider()
