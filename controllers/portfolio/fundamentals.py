import logging
import streamlit as st

from ui.fundamentals import (
    render_fundamental_ranking,
    render_sector_comparison,
)
from shared.errors import AppError


logger = logging.getLogger(__name__)


def render_fundamental_analysis(df_view, tasvc):
    """Render fundamental analysis section."""
    st.subheader("Análisis fundamental del portafolio")
    portfolio_symbols = df_view["simbolo"].tolist()
    if portfolio_symbols:
        with st.spinner("Descargando datos fundamentales…"):
            try:
                fund_df = tasvc.portfolio_fundamentals(portfolio_symbols)
            except AppError as err:
                st.error(str(err))
                return
            except Exception:
                logger.exception("Error al obtener datos fundamentales del portafolio")
                st.error("No se pudieron obtener datos fundamentales, intente más tarde")
                return
        render_fundamental_ranking(fund_df)
        render_sector_comparison(fund_df)
    else:
        st.info("No hay símbolos en el portafolio para analizar.")
