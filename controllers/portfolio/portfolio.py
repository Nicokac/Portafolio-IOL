import logging

import streamlit as st

from domain.models import Controls
from ui.sidebar_controls import render_sidebar
from ui.ui_settings import render_ui_controls
from ui.fundamentals import render_fundamental_data
from ui.export import PLOTLY_CONFIG
from ui.charts import plot_technical_analysis_chart
from ui.favorites import render_favorite_badges, render_favorite_toggle
from application.portfolio_service import PortfolioService, map_to_us_ticker
from application.ta_service import TAService
from application.portfolio_viewmodel import build_portfolio_viewmodel
from shared.errors import AppError
from shared.favorite_symbols import get_persistent_favorites
from services.notifications import NotificationFlags, NotificationsService
from services.portfolio_view import PortfolioViewModelService
from ui.notifications import render_technical_badge, tab_badge_suffix

from .load_data import load_portfolio_data
from .charts import render_basic_section, render_advanced_analysis
from .risk import render_risk_analysis
from .fundamentals import render_fundamental_analysis
logger = logging.getLogger(__name__)

view_model_service = PortfolioViewModelService()
notifications_service = NotificationsService()


def _apply_tab_badges(tab_labels: list[str], flags: NotificationFlags) -> list[str]:
    """Return updated tab labels including badge suffixes for active flags."""

    updated = list(tab_labels)
    if flags.risk_alert and len(updated) > 2:
        suffix = tab_badge_suffix("risk")
        if suffix.strip() and suffix.strip() not in updated[2]:
            updated[2] = f"{updated[2]}{suffix}"
    if flags.upcoming_earnings and len(updated) > 3:
        suffix = tab_badge_suffix("earnings")
        if suffix.strip() and suffix.strip() not in updated[3]:
            updated[3] = f"{updated[3]}{suffix}"
    if flags.technical_signal and len(updated) > 4:
        suffix = tab_badge_suffix("technical")
        if suffix.strip() and suffix.strip() not in updated[4]:
            updated[4] = f"{updated[4]}{suffix}"
    return updated


def render_portfolio_section(container, cli, fx_rates):
    """Render the main portfolio section and return refresh interval."""
    with container:
        psvc = PortfolioService()
        tasvc = TAService()

        df_pos, all_symbols, available_types = load_portfolio_data(cli, psvc)
        favorites = get_persistent_favorites()

        controls: Controls = render_sidebar(all_symbols, available_types)
        render_ui_controls()

        refresh_secs = controls.refresh_secs
        snapshot = view_model_service.get_portfolio_view(
            df_pos=df_pos,
            controls=controls,
            cli=cli,
            psvc=psvc,
        )

        viewmodel = build_portfolio_viewmodel(
            snapshot=snapshot,
            controls=controls,
            fx_rates=fx_rates,
            all_symbols=all_symbols,
        )

        notifications = notifications_service.get_flags()
        tab_labels = _apply_tab_badges(list(viewmodel.tab_options), notifications)

        tab_idx = st.radio(
            "Secciones",
            options=range(len(tab_labels)),
            format_func=lambda i: tab_labels[i],
            horizontal=True,
            key="portfolio_tab",
        )
        controls = viewmodel.controls
        ccl_rate = viewmodel.metrics.ccl_rate
        df_view = viewmodel.positions

        if tab_idx == 0:
            render_basic_section(
                df_view,
                controls,
                ccl_rate,
                favorites=favorites,
                totals=viewmodel.totals,
                historical_total=viewmodel.historical_total,
                contribution_metrics=viewmodel.contributions,
            )
        elif tab_idx == 1:
            render_advanced_analysis(df_view, tasvc)
        elif tab_idx == 2:
            render_risk_analysis(
                df_view,
                tasvc,
                favorites=favorites,
                notifications=notifications,
            )
        elif tab_idx == 3:
            render_fundamental_analysis(
                df_view,
                tasvc,
                favorites=favorites,
                notifications=notifications,
            )
        else:
            st.subheader("Indicadores técnicos por activo")
            if notifications.technical_signal:
                render_technical_badge(
                    help_text="Tenés señales técnicas recientes para revisar en tus activos favoritos.",
                )
            render_favorite_badges(
                favorites,
                empty_message="⭐ Aún no marcaste favoritos para seguimiento rápido.",
            )
            if not all_symbols:
                st.info("No hay símbolos en el portafolio para analizar.")
                return
            all_symbols_vm = list(viewmodel.metrics.all_symbols)
            if not all_symbols_vm:
                st.info("No hay símbolos en el portafolio para analizar.")
            else:
                options = favorites.sort_options(all_symbols_vm)
                if not options:
                    options = all_symbols_vm
                sym = st.selectbox(
                    "Seleccioná un símbolo (CEDEAR / ETF)",
                    options=options,
                    index=favorites.default_index(options),
                    key="ta_symbol",
                    format_func=favorites.format_symbol,
                )
                if sym:
                    render_favorite_toggle(
                        sym,
                        favorites,
                        key_prefix="ta",
                        help_text="Los favoritos quedan disponibles en todas las secciones.",
                    )
                    try:
                        us_ticker = map_to_us_ticker(sym)
                    except ValueError:
                        st.info("No se encontró ticker US para este activo.")
                    else:
                        try:
                            fundamental_data = tasvc.fundamentals(us_ticker) or {}
                        except AppError as err:
                            st.error(str(err))
                        except Exception:
                            logger.exception(
                                "Error al obtener datos fundamentales para %s", sym
                            )
                            st.error(
                                "No se pudieron obtener datos fundamentales, intente más tarde"
                            )
                        else:
                            render_fundamental_data(fundamental_data)

                        cols = st.columns([1, 1, 1, 1])
                        with cols[0]:
                            period = st.selectbox(
                                "Período", ["3mo", "6mo", "1y", "2y"], index=1
                            )
                        with cols[1]:
                            interval = st.selectbox(
                                "Intervalo", ["1d", "1h", "30m"], index=0
                            )
                        with cols[2]:
                            sma_fast = st.number_input(
                                "SMA corta",
                                min_value=5,
                                max_value=100,
                                value=20,
                                step=1,
                            )
                        with cols[3]:
                            sma_slow = st.number_input(
                                "SMA larga",
                                min_value=10,
                                max_value=250,
                                value=50,
                                step=5,
                            )

                        with st.expander("Parámetros adicionales"):
                            c1, c2, c3 = st.columns(3)
                            macd_fast = c1.number_input(
                                "MACD rápida", min_value=5, max_value=50, value=12, step=1
                            )
                            macd_slow = c2.number_input(
                                "MACD lenta", min_value=10, max_value=200, value=26, step=1
                            )
                            macd_signal = c3.number_input(
                                "MACD señal", min_value=5, max_value=50, value=9, step=1
                            )
                            c4, c5, c6 = st.columns(3)
                            atr_win = c4.number_input(
                                "ATR ventana", min_value=5, max_value=200, value=14, step=1
                            )
                            stoch_win = c5.number_input(
                                "Estocástico ventana", min_value=5, max_value=200, value=14, step=1
                            )
                            stoch_smooth = c6.number_input(
                                "Estocástico suavizado", min_value=1, max_value=50, value=3, step=1
                            )
                            c7, c8, c9 = st.columns(3)
                            ichi_conv = c7.number_input(
                                "Ichimoku conv.", min_value=1, max_value=50, value=9, step=1
                            )
                            ichi_base = c8.number_input(
                                "Ichimoku base", min_value=2, max_value=100, value=26, step=1
                            )
                            ichi_span = c9.number_input(
                                "Ichimoku span B", min_value=2, max_value=200, value=52, step=1
                            )

                        try:
                            df_ind = tasvc.indicators_for(
                                sym,
                                period=period,
                                interval=interval,
                                sma_fast=sma_fast,
                                sma_slow=sma_slow,
                                macd_fast=macd_fast,
                                macd_slow=macd_slow,
                                macd_signal=macd_signal,
                                atr_win=atr_win,
                                stoch_win=stoch_win,
                                stoch_smooth=stoch_smooth,
                                ichi_conv=ichi_conv,
                                ichi_base=ichi_base,
                                ichi_span=ichi_span,
                            )
                        except AppError as err:
                            st.error(str(err))
                            return
                        except Exception:
                            logger.exception(
                                "Error al obtener indicadores técnicos para %s", sym
                            )
                            st.error(
                                "No se pudieron obtener indicadores técnicos, intente más tarde"
                            )
                            return
                        if df_ind.empty:
                            st.info(
                                "No se pudo descargar histórico para ese símbolo/periodo/intervalo."
                            )
                        else:
                            fig = plot_technical_analysis_chart(
                                df_ind, sma_fast, sma_slow
                            )
                            st.plotly_chart(
                                fig,
                                width="stretch",
                                key="ta_chart",
                                config=PLOTLY_CONFIG,
                            )
                            st.caption(
                                "Gráfico de precio con indicadores técnicos como "
                                "medias móviles, RSI o MACD para detectar tendencias "
                                "y señales."
                            )
                            alerts = tasvc.alerts_for(df_ind)
                            if alerts:
                                for a in alerts:
                                    al = a.lower()
                                    if "bajista" in al or "sobrecompra" in al:
                                        st.warning(a)
                                    elif "alcista" in al or "sobreventa" in al:
                                        st.success(a)
                                    else:
                                        st.info(a)
                            else:
                                st.caption("Sin alertas técnicas en la última vela.")

                            st.subheader("Backtesting")
                            strat = st.selectbox(
                                "Estrategia", ["SMA", "MACD", "Estocástico", "Ichimoku"], index=0
                            )
                            try:
                                bt = tasvc.backtest(df_ind, strategy=strat)
                            except AppError as err:
                                st.error(str(err))
                                return
                            except Exception:
                                logger.exception(
                                    "Error al ejecutar backtesting para %s", sym
                                )
                                st.error(
                                    "No se pudo ejecutar el backtesting, intente más tarde"
                                )
                                return
                            if bt.empty:
                                st.info("Sin datos suficientes para el backtesting.")
                            else:
                                st.line_chart(bt["equity"])
                                st.caption(
                                    "La línea muestra cómo habría crecido la inversión usando la estrategia seleccionada."
                                )
                                st.metric(
                                    "Retorno acumulado", f"{bt['equity'].iloc[-1] - 1:.2%}"
                                )

        return refresh_secs
