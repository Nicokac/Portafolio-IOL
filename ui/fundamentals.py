# ui\fundamentals.py
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from shared.utils import _is_none_nan_inf, format_number, format_percent

from .export import PLOTLY_CONFIG

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
    "net_margin_ttm": {
        "label": "Margen Neto (TTM)",
        "format": format_percent,
        "desc": "Margen neto de los últimos 12 meses según FMP.",
        "url": "https://site.financialmodelingprep.com/developer/docs/ratios-api/",
    },
    "gross_margin": {
        "label": "Margen Bruto (TTM)",
        "format": format_percent,
        "desc": "Relación entre la ganancia bruta y los ingresos (últimos 12 meses).",
        "url": "https://site.financialmodelingprep.com/developer/docs/ratios-api/",
    },
    "return_on_assets": {
        "label": "ROA",
        "format": format_percent,
        "desc": "Retorno sobre los activos totales promedio.",
        "url": "https://www.investopedia.com/terms/r/returnonassets.asp",
    },
    "operating_margin": {
        "label": "Margen Operativo",
        "format": format_percent,
        "desc": "Ganancia operativa como porcentaje de los ingresos.",
        "url": "https://www.investopedia.com/terms/o/operatingmargin.asp",
    },
    "ebitda_margin": {
        "label": "Margen EBITDA (TTM)",
        "format": format_percent,
        "desc": "Margen EBITDA de los últimos 12 meses según FMP.",
        "url": "https://site.financialmodelingprep.com/developer/docs/ratios-api/",
    },
    "fcf_yield": {
        "label": "FCF Yield",
        "format": format_percent,
        "desc": "Rendimiento del flujo de caja libre sobre el valor de la empresa.",
        "url": "https://www.investopedia.com/terms/f/free-cash-flow-yield.asp",
    },
    "interest_coverage": {
        "label": "Cobertura de Intereses",
        "format": lambda v: "—" if _is_none_nan_inf(v) else f"{float(v):.2f}×",
        "desc": "Cuántas veces la ganancia operativa cubre los gastos financieros.",
        "url": "https://www.investopedia.com/terms/i/interestcoverageratio.asp",
    },
    "debt_to_equity": {
        "label": "Deuda/Patrimonio",
        "format": lambda v: "—" if _is_none_nan_inf(v) else f"{float(v):.2f}",
        "desc": "Proporción de financiamiento con deuda respecto al capital propio.",
        "url": "https://www.investopedia.com/terms/d/debtequityratio.asp",
    },
    "debt_to_ebitda": {
        "label": "Deuda/EBITDA",
        "format": lambda v: "—" if _is_none_nan_inf(v) else f"{float(v):.2f}×",
        "desc": "Relación entre deuda neta y EBITDA (últimos 12 meses).",
        "url": "https://site.financialmodelingprep.com/developer/docs/ratios-api/",
    },
    "payout_ratio": {
        "label": "Payout Ratio (TTM)",
        "format": format_percent,
        "desc": "Porcentaje de ganancias destinado a dividendos en 12 meses.",
        "url": "https://www.investopedia.com/terms/p/payoutratio.asp",
    },
    "quick_ratio": {
        "label": "Razón Rápida",
        "format": lambda v: "—" if _is_none_nan_inf(v) else f"{float(v):.2f}",
        "desc": "Liquidez inmediata sobre pasivos corrientes (últimos 12 meses).",
        "url": "https://www.investopedia.com/terms/q/quickratio.asp",
    },
    "current_ratio": {
        "label": "Razón Corriente",
        "format": lambda v: "—" if _is_none_nan_inf(v) else f"{float(v):.2f}",
        "desc": "Liquidez corriente total respecto a pasivos corrientes.",
        "url": "https://www.investopedia.com/terms/c/currentratio.asp",
    },
}


def render_fundamental_data(data: dict):
    if not data or (isinstance(data, dict) and data.get("error")):
        if isinstance(data, dict):
            st.warning(data.get("error", "Datos fundamentales no disponibles."))
        else:
            st.warning("Datos fundamentales no disponibles.")
        return

    st.subheader(f"Análisis Fundamental: {data.get('name', '—')}")
    st.caption(f"**Sector:** {data.get('sector', '—')} | **Web:** {data.get('website', '—')}")

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
    st.caption("Resumen de indicadores fundamentales básicos.")
    st.divider()


def render_fundamental_ranking(df: pd.DataFrame):
    """Muestra ranking y filtros por métricas fundamentales/ESG."""
    if df is None or df.empty:
        st.info("No se pudieron obtener datos fundamentales.")
        return

    sectors = sorted([s for s in df["sector"].dropna().unique()])
    sector = st.selectbox("Sector", ["Todos"] + sectors)
    if sector != "Todos":
        df = df[df["sector"] == sector]

    metric = st.selectbox(
        "Ordenar por",
        [
            "market_cap",
            "pe_ratio",
            "revenue_growth",
            "earnings_growth",
            "gross_margin",
            "net_margin_ttm",
            "return_on_assets",
            "operating_margin",
            "ebitda_margin",
            "fcf_yield",
            "interest_coverage",
            "debt_to_ebitda",
            "payout_ratio",
            "quick_ratio",
            "current_ratio",
            "esg_score",
        ],
        index=0,
    )
    df_sorted = df.sort_values(by=metric, ascending=False)
    st.dataframe(df_sorted.reset_index(drop=True))
    st.caption("Ordena las empresas según la métrica elegida para comparar de forma rápida.")

    neg_growth = df_sorted[df_sorted["earnings_growth"].notna() & (df_sorted["earnings_growth"] < 0)]
    if not neg_growth.empty:
        st.warning("Alerta: crecimiento de ganancias negativo en algunos activos.")

    low_esg = df_sorted[df_sorted["esg_score"].notna() & (df_sorted["esg_score"] < 30)]
    if not low_esg.empty:
        st.warning("Alerta ESG: puntajes ESG bajos detectados.")


def render_sector_comparison(df: pd.DataFrame):
    """Graficar métricas comparadas contra el promedio del sector."""
    if df is None or df.empty:
        return
    st.subheader("Comparativa vs promedio del sector")
    metrics = [
        "pe_ratio",
        "price_to_book",
        "return_on_equity",
        "profit_margin",
        "net_margin_ttm",
        "gross_margin",
        "return_on_assets",
        "operating_margin",
        "ebitda_margin",
        "fcf_yield",
        "interest_coverage",
        "debt_to_equity",
        "debt_to_ebitda",
        "payout_ratio",
        "quick_ratio",
        "current_ratio",
    ]
    metric = st.selectbox("Métrica", metrics)
    dfm = df[df[metric].notna()].copy()
    if dfm.empty:
        st.info("No hay datos disponibles para la métrica seleccionada.")
        return
    dfm["sector_avg"] = dfm.groupby("sector")[metric].transform("mean")
    dfm["rel"] = dfm[metric] / dfm["sector_avg"]
    fig = px.bar(dfm, x="symbol", y="rel", color="sector", labels={"rel": "Ratio vs sector"})
    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="red",
        annotation_text="Promedio sector",
        annotation_position="top left",
    )
    st.plotly_chart(fig, config=PLOTLY_CONFIG)
    st.caption("Valores mayores a 1 indican métricas por encima del promedio del sector (posible sobrevaluación).")
