import logging
import time

import streamlit as st

from shared.favorite_symbols import FavoriteSymbols, get_persistent_favorites
from ui.favorites import render_favorite_badges, render_favorite_toggle
from ui.fundamentals import (
    render_fundamental_ranking,
    render_sector_comparison,
)
from services.notifications import NotificationFlags
from shared.errors import AppError
from ui.notifications import render_earnings_badge
from services.health import record_tab_latency


logger = logging.getLogger(__name__)


def render_fundamental_analysis(
    df_view,
    tasvc,
    favorites: FavoriteSymbols | None = None,
    *,
    notifications: NotificationFlags | None = None,
):
    """Render fundamental analysis section."""
    favorites = favorites or get_persistent_favorites()
    st.subheader("Análisis fundamental del portafolio")
    if notifications and notifications.upcoming_earnings:
        render_earnings_badge(
            help_text="Algunas empresas de tu portafolio reportan resultados en los próximos días.",
        )
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
        latency_ms: float | None = None
        with st.spinner("Descargando datos fundamentales…"):
            start_time = time.perf_counter()
            try:
                fund_df = tasvc.portfolio_fundamentals(portfolio_symbols)
            except AppError as err:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                record_tab_latency("fundamentales", latency_ms, status="error")
                st.error(str(err))
                return
            except Exception:
                logger.exception("Error al obtener datos fundamentales del portafolio")
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                record_tab_latency("fundamentales", latency_ms, status="error")
                st.error("No se pudieron obtener datos fundamentales, intente más tarde")
                return
            latency_ms = (time.perf_counter() - start_time) * 1000.0
        record_tab_latency("fundamentales", latency_ms, status="success")
        render_fundamental_ranking(fund_df)
        render_sector_comparison(fund_df)
    else:
        st.info("No hay símbolos en el portafolio para analizar.")
