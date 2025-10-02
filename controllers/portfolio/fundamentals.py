import logging
import streamlit as st

from shared.favorite_symbols import FavoriteSymbols, get_persistent_favorites
from ui.favorites import render_favorite_badges, render_favorite_toggle
from ui.fundamentals import (
    render_fundamental_ranking,
    render_sector_comparison,
)
from shared.errors import AppError


logger = logging.getLogger(__name__)


def render_fundamental_analysis(df_view, tasvc, favorites: FavoriteSymbols | None = None):
    """Render fundamental analysis section."""
    favorites = favorites or get_persistent_favorites()
    st.subheader("Análisis fundamental del portafolio")
    symbols = (
        sorted({str(sym) for sym in df_view.get("simbolo", []) if str(sym).strip()})
        if not df_view.empty
        else []
    )

    render_favorite_badges(
        favorites,
        empty_message="⭐ Marcá favoritos para enfocarte en los activos clave al analizar fundamentos.",
    )
    if symbols:
        options = favorites.sort_options(symbols)
        selected_symbol = st.selectbox(
            "Gestionar favoritos",
            options=options,
            index=favorites.default_index(options),
            key="fundamental_favorite_select",
            format_func=favorites.format_symbol,
        )
        render_favorite_toggle(
            selected_symbol,
            favorites,
            key_prefix="fundamental",
            help_text="Los favoritos se comparten entre todas las pestañas y exportaciones.",
        )

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
