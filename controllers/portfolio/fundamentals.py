import streamlit as st

from ui.fundamentals import (
    render_fundamental_ranking,
    render_sector_comparison,
)


def render_fundamental_analysis(df_view, tasvc):
    """Render fundamental analysis section."""
    st.subheader("Análisis fundamental del portafolio")
    portfolio_symbols = df_view["simbolo"].tolist()
    if portfolio_symbols:
        with st.spinner("Descargando datos fundamentales…"):
            fund_df = tasvc.portfolio_fundamentals(portfolio_symbols)
        render_fundamental_ranking(fund_df)
        render_sector_comparison(fund_df)
    else:
        st.info("No hay símbolos en el portafolio para analizar.")
