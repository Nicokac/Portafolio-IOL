import logging
import time

import pandas as pd
import streamlit as st

from services.cache.market_data_cache import get_market_data_cache
from services.health import record_tab_latency
from services.notifications import NotificationFlags
from shared.errors import AppError
from shared.favorite_symbols import FavoriteSymbols, get_persistent_favorites
from ui.favorites import render_favorite_badges, render_favorite_toggle
from ui.fundamentals import (
    render_fundamental_ranking,
    render_sector_comparison,
)
from ui.notifications import render_earnings_badge

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
    symbols = sorted({str(sym) for sym in df_view.get("simbolo", []) if str(sym).strip()}) if not df_view.empty else []

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
                cache = get_market_data_cache()
                sector_hints = sorted({str(val).strip() for val in df_view.get("tipo", []) if str(val).strip()})
                fund_df = cache.get_fundamentals(
                    portfolio_symbols,
                    loader=lambda symbols=list(portfolio_symbols): tasvc.portfolio_fundamentals(list(symbols)),
                    sectors=sector_hints,
                )
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
        if isinstance(fund_df, pd.DataFrame) and not fund_df.empty:
            for col in fund_df.columns:
                if fund_df[col].dtype == object:
                    fund_df[col] = pd.to_numeric(fund_df[col], errors="coerce")
            float_cols = [col for col in fund_df.columns if pd.api.types.is_float_dtype(fund_df[col])]
            if float_cols:
                fund_df[float_cols] = fund_df[float_cols].astype("float32")
        render_fundamental_ranking(fund_df)
        render_sector_comparison(fund_df)
    else:
        st.info("No hay símbolos en el portafolio para analizar.")
