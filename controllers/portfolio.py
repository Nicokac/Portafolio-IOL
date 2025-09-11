import time
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from domain.models import Controls
from shared.config import settings
from ui.tables import render_totals, render_table
from ui.sidebar_controls import render_sidebar
from ui.ui_settings import render_ui_controls
from ui.charts import (
    plot_pl_topn,
    plot_donut_tipo,
    plot_dist_por_tipo,
    plot_bubble_pl_vs_costo,
    plot_heat_pl_pct,
    plot_pl_daily_topn,
    plot_correlation_heatmap,
    plot_technical_analysis_chart,
)
from ui.fundamentals import (
    render_fundamental_data,
    render_fundamental_ranking,
    render_sector_comparison,
)
from ui.export import PLOTLY_CONFIG
from application.portfolio_service import PortfolioService, map_to_us_ticker
from application.ta_service import TAService
from application.risk_service import (
    compute_returns,
    annualized_volatility,
    beta,
    historical_var,
    markowitz_optimize,
    monte_carlo_simulation,
    apply_stress,
)
from services.cache import fetch_portfolio, fetch_quotes_bulk


def _load_portfolio_data(cli, psvc):
    """Fetch and normalize portfolio positions."""
    with st.spinner("Cargando y actualizando portafolio... ⏳"):
        try:
            payload = fetch_portfolio(cli)
        except Exception as e:  # pragma: no cover - streamlit error path
            st.error(f"Error al consultar portafolio: {e}")
            st.stop()

    if isinstance(payload, dict) and payload.get("_cached"):
        st.warning("No se pudo contactar a IOL; mostrando datos del portafolio en caché.")

    if isinstance(payload, dict) and "message" in payload:
        st.info(f"ℹ️ Mensaje de IOL: \"{payload['message']}\"")
        st.stop()

    df_pos = psvc.normalize_positions(payload)
    if df_pos.empty:
        st.warning("No se encontraron posiciones o no pudimos mapear la respuesta.")
        if isinstance(payload, dict) and "activos" in payload:
            st.dataframe(pd.DataFrame(payload["activos"]).head(20))
        st.stop()

    all_symbols = sorted(df_pos["simbolo"].astype(str).str.upper().unique())
    available_types = sorted(
        {
            psvc.classify_asset_cached(s)
            for s in all_symbols
            if psvc.classify_asset_cached(s)
        }
    )
    return df_pos, all_symbols, available_types


def _apply_filters(df_pos, controls, cli, psvc):
    """Apply user filters and enrich positions with quotes."""
    if controls.hide_cash:
        df_pos = df_pos[~df_pos["simbolo"].isin(["IOLPORA", "PARKING"])].copy()
    if controls.selected_syms:
        df_pos = df_pos[df_pos["simbolo"].isin(controls.selected_syms)].copy()

    pairs = list(
        df_pos[["mercado", "simbolo"]]
        .drop_duplicates()
        .astype({"mercado": str, "simbolo": str})
        .itertuples(index=False, name=None)
    )
    quotes_map = fetch_quotes_bulk(cli, pairs)

    df_view = psvc.calc_rows(
        lambda mercado, simbolo=None: quotes_map.get(
            (str(mercado).lower(), str((simbolo or mercado)).upper()), {}
        ),
        df_pos,
        exclude_syms=[],
    )
    if df_view.empty:
        return df_view

    df_view["tipo"] = df_view["simbolo"].astype(str).map(psvc.classify_asset_cached)

    if controls.selected_types:
        df_view = df_view[df_view["tipo"].isin(controls.selected_types)].copy()

    symbol_q = (controls.symbol_query or "").strip()
    if symbol_q:
        df_view = df_view[
            df_view["simbolo"].astype(str).str.contains(symbol_q, case=False, na=False)
        ].copy()

    chg_map = {k: v.get("chg_pct") for k, v in quotes_map.items()}
    map_keys = df_view.apply(
        lambda row: (str(row["mercado"]).lower(), str(row["simbolo"]).upper()), axis=1
    )

    df_view["chg_%"] = map_keys.map(chg_map)
    df_view["chg_%"] = pd.to_numeric(df_view["chg_%"], errors="coerce")

    st.session_state.setdefault("quotes_hist", {})
    now_ts = int(time.time())
    for (mkt, sym), chg in chg_map.items():
        if isinstance(chg, (int, float)):
            st.session_state["quotes_hist"].setdefault(sym, [])
            if (
                not st.session_state["quotes_hist"][sym]
                or (st.session_state["quotes_hist"][sym][-1].get("ts") != now_ts)
            ):
                st.session_state["quotes_hist"][sym].append({"ts": now_ts, "chg_pct": float(chg)})
                maxlen = getattr(settings, "quotes_hist_maxlen", 500)
                st.session_state["quotes_hist"][sym] = st.session_state["quotes_hist"][sym][-maxlen:]

    return df_view


def _generate_basic_charts(df_view, top_n):
    """Generate basic portfolio charts."""
    return {
        "pl_topn": plot_pl_topn(df_view, n=top_n),
        "donut_tipo": plot_donut_tipo(df_view),
        "dist_tipo": plot_dist_por_tipo(df_view),
        "pl_diario": plot_pl_daily_topn(df_view, n=top_n),
    }


def _compute_risk_metrics(returns_df, bench_ret, weights):
    """Compute core risk metrics for the portfolio."""
    port_ret = returns_df.mul(weights, axis=1).sum(axis=1)
    vol = annualized_volatility(port_ret)
    b = beta(port_ret, bench_ret)
    var_95 = historical_var(port_ret)
    opt_w = markowitz_optimize(returns_df)
    return vol, b, var_95, opt_w, port_ret


def _render_basic_section(df_view, controls, ccl_rate):
    """Render totals, table and basic charts for the portfolio."""
    if df_view.empty:
        st.info("No hay datos del portafolio para mostrar.")
        return

    render_totals(df_view, ccl_rate=ccl_rate)
    render_table(
        df_view,
        controls.order_by,
        controls.desc,
        ccl_rate=ccl_rate,
        show_usd=controls.show_usd,
    )

    charts = _generate_basic_charts(df_view, controls.top_n)
    colA, colB = st.columns(2)
    with colA:
        st.subheader("P/L por símbolo (Top N)")
        fig = charts["pl_topn"]
        if fig is not None:
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="pl_topn",
                config=PLOTLY_CONFIG,
            )
        else:
            st.info("Sin datos para graficar P/L Top N.")
    with colB:
        st.subheader("Composición por tipo (Donut)")
        fig = charts["donut_tipo"]
        if fig is not None:
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="donut_tipo",
                config=PLOTLY_CONFIG,
            )
        else:
            st.info("No hay datos para el donut por tipo.")

    st.subheader("Distribución por tipo (Valorizado)")
    fig = charts["dist_tipo"]
    if fig is not None:
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="dist_tipo",
            config=PLOTLY_CONFIG,
        )
    else:
        st.info("No hay datos para la distribución por tipo.")

    st.subheader("P/L diario por símbolo (Top N)")
    fig = charts["pl_diario"]
    if fig is not None:
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="pl_diario",
            config=PLOTLY_CONFIG,
        )
    else:
        st.info("Aún no hay datos de P/L diario.")


def _render_advanced_analysis(df_view):
    """Render advanced analysis charts (bubble and heatmap)."""
    st.subheader("Bubble Chart Interactivo")
    axis_options = [
        c
        for c in ["costo", "pl", "pl_%", "valor_actual", "pl_d"]
        if c in df_view.columns
    ]
    if not axis_options:
        st.info("No hay columnas disponibles para el gráfico bubble.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            x_axis = st.selectbox(
                "Eje X",
                options=axis_options,
                index=axis_options.index("costo") if "costo" in axis_options else 0,
                key="bubble_x",
            )
            x_log = st.checkbox("Escala log X", key="bubble_x_log")
        with c2:
            y_axis = st.selectbox(
                "Eje Y",
                options=axis_options,
                index=axis_options.index("pl")
                if "pl" in axis_options
                else min(1, len(axis_options) - 1),
                key="bubble_y",
            )
            y_log = st.checkbox("Escala log Y", key="bubble_y_log")
        palette_opt = st.selectbox(
            "Paleta",
            ["Tema", "Plotly", "D3", "G10"],
            key="bubble_palette",
        )
        palette_map = {
            "Plotly": px.colors.qualitative.Plotly,
            "D3": px.colors.qualitative.D3,
            "G10": px.colors.qualitative.G10,
        }
        color_seq = palette_map.get(palette_opt) if palette_opt != "Tema" else None
        fig = plot_bubble_pl_vs_costo(
            df_view,
            x_axis=x_axis,
            y_axis=y_axis,
            color_seq=color_seq,
            log_x=x_log,
            log_y=y_log,
        )
        if fig is not None:
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="bubble_chart",
                config=PLOTLY_CONFIG,
            )
            with st.expander("Descripción"):
                st.caption(
                    "Cada burbuja representa un símbolo; el tamaño refleja el valor actual. Cambia ejes, escalas y paleta para explorar distintos ángulos."
                )
        else:
            st.info("No hay datos suficientes para el gráfico bubble.")

        st.subheader("Heatmap de rendimiento (%) por símbolo")
        heat_scale = st.selectbox(
            "Escala de color",
            ["RdBu", "Viridis", "Plasma", "Cividis", "Turbo"],
            key="heat_scale",
        )
        fig = plot_heat_pl_pct(df_view, color_scale=heat_scale)
        if fig is not None:
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="heatmap_chart",
                config=PLOTLY_CONFIG,
            )
            with st.expander("Descripción"):
                st.caption(
                    "El color indica la variación porcentual; prueba diferentes escalas para resaltar ganancias o pérdidas."
                )
        else:
            st.info("No hay datos suficientes para el heatmap.")


def _render_risk_analysis(df_view, tasvc):
    """Render correlation and risk analysis for the portfolio."""
    st.subheader("Análisis de Correlación del Portafolio")
    corr_period = st.selectbox(
        "Calcular correlación sobre el último período:",
        ["3mo", "6mo", "1y", "2y", "5y"],
        index=2,
    )
    portfolio_symbols = df_view["simbolo"].tolist()
    if len(portfolio_symbols) >= 2:
        with st.spinner(f"Calculando correlación ({corr_period})…"):
            hist_df = tasvc.portfolio_history(
                simbolos=portfolio_symbols, period=corr_period
            )
        fig = plot_correlation_heatmap(hist_df)
        if fig:
            st.caption(
                """
                Un heatmap de correlación muestra cómo se mueven los activos entre sí.
                **Azul (cercano a 1)**: Se mueven juntos.
                **Rojo (cercano a -1)**: Se mueven en direcciones opuestas.
                **Blanco (cercano a 0)**: No tienen relación.
                Una buena diversificación busca valores bajos (cercanos a 0 o negativos).
                """
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="corr_heatmap",
                config=PLOTLY_CONFIG,
            )
        else:
            st.warning(
                f"No se pudieron obtener suficientes datos históricos para el período '{corr_period}' para calcular la correlación."
            )
    else:
        st.info(
            "Necesitas al menos 2 activos en tu portafolio (después de aplicar filtros) para calcular la correlación."
        )

    st.subheader("Análisis de Riesgo")
    if portfolio_symbols:
        with st.spinner("Descargando históricos…"):
            prices_df = tasvc.portfolio_history(
                simbolos=portfolio_symbols, period="1y"
            )
            bench_df = tasvc.portfolio_history(simbolos=["^GSPC"], period="1y")
        if prices_df.empty or bench_df.empty:
            st.info(
                "No se pudieron obtener datos históricos para calcular métricas de riesgo."
            )
        else:
            returns_df = compute_returns(prices_df)
            bench_ret = compute_returns(bench_df).squeeze()
            weights = (
                df_view.set_index("simbolo")["valor_actual"].astype(float)
                .reindex(returns_df.columns)
                .dropna()
            )
            weights = weights / weights.sum() if not weights.empty else weights
            if weights.empty or returns_df.empty:
                st.info(
                    "No hay suficientes datos para calcular métricas de riesgo."
                )
            else:
                vol, b, var_95, opt_w, port_ret = _compute_risk_metrics(
                    returns_df, bench_ret, weights
                )

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Volatilidad anualizada",
                    f"{vol:.2%}" if vol == vol else "N/A",
                )
                c2.metric(
                    "Beta vs S&P 500",
                    f"{b:.2f}" if b == b else "N/A",
                )
                c3.metric(
                    "VaR 5%",
                    f"{var_95:.2%}" if var_95 == var_95 else "N/A",
                )

                with st.expander("Volatilidad - evolución"):
                    rolling_vol = port_ret.rolling(30).std() * np.sqrt(252)
                    fig_vol = px.line(
                        rolling_vol,
                        labels={
                            "index": "Fecha",
                            "value": "Volatilidad anualizada",
                        },
                    )
                    st.plotly_chart(
                        fig_vol,
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )
                    st.caption(
                        "La volatilidad refleja la variabilidad de los retornos; aquí se muestra en una ventana móvil de 30 días."
                    )

                with st.expander("Distribución de retornos y VaR"):
                    var_threshold = np.quantile(port_ret, 0.05)
                    fig_var = px.histogram(port_ret, nbins=50)
                    fig_var.add_vline(
                        x=var_threshold,
                        line_color="red",
                        annotation_text="VaR 5%",
                        annotation_position="top left",
                    )
                    st.plotly_chart(
                        fig_var,
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )
                    st.caption(
                        "La línea roja indica el VaR al 5%, representando la pérdida máxima esperada con 95% de confianza."
                    )

                with st.expander(
                    "Optimización de portafolio (Markowitz)"
                ):
                    opt_df = pd.DataFrame(
                        {"ticker": opt_w.index, "weight": opt_w.values}
                    )
                    st.bar_chart(opt_df, x="ticker", y="weight")

                with st.expander("Simulación Monte Carlo"):
                    sims = st.number_input(
                        "Nº de simulaciones",
                        min_value=100,
                        max_value=10000,
                        value=1000,
                        step=100,
                    )
                    horizon = st.number_input(
                        "Horizonte (días)",
                        min_value=30,
                        max_value=365,
                        value=252,
                        step=30,
                    )
                    final_prices = monte_carlo_simulation(
                        returns_df, weights, n_sims=sims, horizon=horizon
                    )
                    st.line_chart(final_prices)

                with st.expander("Aplicar shocks"):
                    templates = {"Leve": 0.03, "Moderado": 0.07, "Fuerte": 0.12}
                    tmpl = st.selectbox("Escenario", list(templates), index=0)
                    shocks = {
                        sym: -templates[tmpl] for sym in returns_df.columns
                    }
                    st.caption(
                        f"Aplicando un shock uniforme de {templates[tmpl]:.0%} a todos los activos."
                    )
                base_prices = pd.Series(1.0, index=weights.index)
                stressed_val = apply_stress(base_prices, weights, shocks)
                st.write(
                    f"Retorno con shocks: {stressed_val - 1:.2%}"
                )
    else:
        st.info("No hay símbolos en el portafolio para analizar.")


def _render_fundamental_analysis(df_view, tasvc):
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

def render_portfolio_section(container, cli, fx_rates):
    """Render the main portfolio section and return refresh interval."""
    with container:
        psvc = PortfolioService()
        tasvc = TAService()

        df_pos, all_symbols, available_types = _load_portfolio_data(cli, psvc)

        controls: Controls = render_sidebar(all_symbols, available_types)
        render_ui_controls()

        refresh_secs = controls.refresh_secs

        df_view = _apply_filters(df_pos, controls, cli, psvc)

        ccl_rate = fx_rates.get("ccl")

        if "portfolio_tab" not in st.session_state:
            st.session_state["portfolio_tab"] = 0

        tabs = st.tabs(
            [
                "📂 Portafolio",
                "📊 Análisis avanzado",
                "🎲 Análisis de Riesgo",
                "📑 Análisis fundamental",
                "🔎 Análisis de activos",
            ],
            key="portfolio_tab",
        )

        with tabs[0]:
            _render_basic_section(df_view, controls, ccl_rate)
        with tabs[1]:
            _render_advanced_analysis(df_view)
        with tabs[2]:
            _render_risk_analysis(df_view, tasvc)
        with tabs[3]:
            _render_fundamental_analysis(df_view, tasvc)

        # Pestaña 5
        with tabs[4]:
            st.subheader("Indicadores técnicos por activo")
            if not all_symbols:
                st.info("No hay símbolos en el portafolio para analizar.")
            else:
                sym = st.selectbox(
                    "Seleccioná un símbolo (CEDEAR / ETF)",
                    options=all_symbols,
                    index=0,
                    key="ta_symbol",
                )
                if sym:
                    us_ticker = map_to_us_ticker(sym)
                    if not us_ticker:
                        st.info("No se encontró ticker US para este activo.")
                    else:
                        fundamental_data = tasvc.fundamentals(us_ticker) or {}
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
                                use_container_width=True,
                                key="ta_chart",
                                config=PLOTLY_CONFIG,
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
                            bt = tasvc.backtest(df_ind, strategy=strat)
                            if bt.empty:
                                st.info("Sin datos suficientes para el backtesting.")
                            else:
                                st.line_chart(bt["equity"])
                                st.metric(
                                    "Retorno acumulado", f"{bt['equity'].iloc[-1] - 1:.2%}"
                                )

        return refresh_secs
