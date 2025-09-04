# ui\fundamentals.py
from __future__ import annotations

import pandas as pd
import streamlit as st

from shared.utils import _is_none_nan_inf, format_number, format_percent

# Meta información de indicadores: etiqueta, formato, descripción y fuente
INDICATORS = {
    "market_cap": {
        "label": "Capitalización de Mercado",
        "format": lambda v: f"US$ {format_number(v)}",
        "desc": "Valor total de las acciones en circulación.",
        "url": "https://es.wikipedia.org/wiki/Capitalizaci%C3%B3n_burs%C3%A1til",
    },
    "pe_ratio": {
        "label": "Ratio P/E (TTM)",
        "format": lambda v: "—" if _is_none_nan_inf(v) else f"{float(v):.2f}",
        "desc": "Relación precio/beneficio de los últimos doce meses.",
        "url": "https://www.investopedia.com/terms/p/price-earningsratio.asp",
    },
    "dividend_yield": {
        "label": "Rendimiento por Dividendo",
        "format": format_percent,
        "desc": "Porcentaje de retorno anual por dividendos.",
        "url": "https://www.investopedia.com/terms/d/dividendyield.asp",
    },
    "price_to_book": {
        "label": "Precio/Valor Libro",
        "format": lambda v: "—" if _is_none_nan_inf(v) else f"{float(v):.2f}",
        "desc": "Compara la capitalización con el patrimonio neto contable.",
        "url": "https://www.investopedia.com/terms/p/price-to-bookratio.asp",
    },
    "return_on_equity": {
        "label": "ROE",
        "format": format_percent,
        "desc": "Retorno sobre el patrimonio promedio.",
        "url": "https://www.investopedia.com/terms/r/returnonequity.asp",
    },
    "profit_margin": {
        "label": "Margen Neto",
        "format": format_percent,
        "desc": "Porcentaje de ganancia neta sobre las ventas.",
        "url": "https://www.investopedia.com/terms/p/profitmargin.asp",
    },
    "debt_to_equity": {
        "label": "Deuda/Patrimonio",
        "format": lambda v: "—" if _is_none_nan_inf(v) else f"{float(v):.2f}",
        "desc": "Proporción de financiamiento con deuda respecto al capital propio.",
        "url": "https://www.investopedia.com/terms/d/debtequityratio.asp",
    },
}

def render_fundamental_data(data: dict):
    if not data or data.get("error"):
        st.warning(data.get("error", "Datos fundamentales no disponibles."))
        return

    st.subheader(f"Análisis Fundamental: {data.get('name', '—')}")
    st.caption(f"**Sector:** {data.get('sector', '—')} | **Web:** {data.get('website', '—')}")

    # col1, col2, col3 = st.columns(3)
    # col1.metric("Capitalización de Mercado", f"US$ {format_number(data.get('market_cap'))}")

    # pe = data.get("pe_ratio")
    # col2.metric("Ratio P/E (TTM)", "—" if _is_none_nan_inf(pe) else f"{float(pe):.2f}")

    # col3.metric("Rendimiento por Dividendo", format_percent(data.get("dividend_yield")))

    rows = []
    for key, meta in INDICATORS.items():
        val = data.get(key)
        formatted = meta["format"](val)
        rows.append(
            {
                "Indicador": meta["label"],
                "Valor": formatted,
                "Descripción": meta["desc"],
                "Fuente": f"[Link]({meta['url']})",
            }
        )
        
    st.table(pd.DataFrame(rows))
    st.divider()
