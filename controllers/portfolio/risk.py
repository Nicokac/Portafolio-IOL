import logging

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from shared.favorite_symbols import FavoriteSymbols, get_persistent_favorites
from ui.favorites import render_favorite_badges, render_favorite_toggle
from application.risk_service import (
    compute_returns,
    annualized_volatility,
    beta,
    historical_var,
    expected_shortfall,
    rolling_correlations,
    markowitz_optimize,
    monte_carlo_simulation,
    apply_stress,
    asset_risk_breakdown,
    max_drawdown,
    drawdown_series,
)
from ui.charts import plot_correlation_heatmap, _apply_layout
from ui.export import PLOTLY_CONFIG
from shared.errors import AppError


logger = logging.getLogger(__name__)


def compute_risk_metrics(returns_df, bench_ret, weights, *, var_confidence: float = 0.95):
    """Compute core risk metrics for the portfolio."""
    port_ret = returns_df.mul(weights, axis=1).sum(axis=1)
    vol = annualized_volatility(port_ret)
    b = beta(port_ret, bench_ret)
    var_value = historical_var(port_ret, confidence=var_confidence)
    cvar_value = expected_shortfall(port_ret, confidence=var_confidence)
    opt_w = markowitz_optimize(returns_df)
    asset_vols, asset_drawdowns = asset_risk_breakdown(returns_df)
    port_drawdown = max_drawdown(port_ret)
    return (
        vol,
        b,
        var_value,
        cvar_value,
        opt_w,
        port_ret,
        asset_vols,
        asset_drawdowns,
        port_drawdown,
    )


def render_risk_analysis(df_view, tasvc, favorites: FavoriteSymbols | None = None):
    """Render correlation and risk analysis for the portfolio."""
    favorites = favorites or get_persistent_favorites()
    st.subheader("Análisis de Correlación del Portafolio")
    symbols = (
        sorted({str(sym) for sym in df_view.get("simbolo", []) if str(sym).strip()})
        if not df_view.empty
        else []
    )

    render_favorite_badges(
        favorites,
        empty_message="⭐ Marcá tus favoritos para seguirlos de cerca en el análisis de riesgo.",
    )
    if symbols:
        options = favorites.sort_options(symbols)
        selected_symbol = st.selectbox(
            "Gestionar favoritos",
            options=options,
            index=favorites.default_index(options),
            key="risk_favorite_select",
            format_func=favorites.format_symbol,
        )
        render_favorite_toggle(
            selected_symbol,
            favorites,
            key_prefix="risk",
            help_text="Tus favoritos se sincronizan entre portafolio, riesgo, técnico y fundamental.",
        )

    corr_period = st.selectbox(
        "Calcular correlación sobre el último período:",
        ["3mo", "6mo", "1y", "2y", "5y"],
        index=2,
    )
    portfolio_symbols = df_view["simbolo"].tolist()
    if len(portfolio_symbols) >= 2:
        with st.spinner(f"Calculando correlación ({corr_period})…"):
            try:
                hist_df = tasvc.portfolio_history(
                    simbolos=portfolio_symbols, period=corr_period
                )
            except AppError as err:
                st.error(str(err))
                return
            except Exception:
                logger.exception(
                    "Error al obtener históricos para correlación",
                )
                st.error(
                    "No se pudieron obtener datos históricos, intente nuevamente más tarde",
                )
                return
        returns_for_corr = compute_returns(hist_df)
        fig = plot_correlation_heatmap(hist_df)
        if fig:
            st.plotly_chart(
                fig,
                width="stretch",
                key="corr_heatmap",
                config=PLOTLY_CONFIG,
            )
            st.caption(
                """
                Un heatmap de correlación muestra cómo se mueven los activos entre sí.
                **Azul (cercano a 1)**: Se mueven juntos.
                **Rojo (cercano a -1)**: Se mueven en direcciones opuestas.
                **Blanco (cercano a 0)**: No tienen relación.
                Una buena diversificación busca valores bajos (cercanos a 0 o negativos).
                """
            )
            if returns_for_corr.shape[1] >= 2:
                window_options = {"1 mes (21)": 21, "3 meses (63)": 63, "6 meses (126)": 126}
                selected_window_label = st.selectbox(
                    "Ventana para correlaciones móviles",
                    list(window_options.keys()),
                    index=1,
                    key="rolling_corr_window",
                )
                selected_window = window_options[selected_window_label]
                rolling_df = rolling_correlations(returns_for_corr, selected_window)
                if not rolling_df.empty:
                    roll_fig = px.line(
                        rolling_df,
                        labels={"index": "Fecha", "value": "Correlación", "variable": "Par"},
                    )
                    roll_fig = _apply_layout(
                        roll_fig,
                        title=f"Correlaciones móviles ({selected_window} ruedas)",
                    )
                    st.plotly_chart(
                        roll_fig,
                        width="stretch",
                        key="rolling_corr_chart",
                        config=PLOTLY_CONFIG,
                    )
                    latest = rolling_df.dropna(how="all").tail(1)
                    if not latest.empty:
                        latest_tidy = (
                            latest.T.reset_index().rename(columns={"index": "Par", latest.index[-1]: "Correlación"})
                        )
                        st.markdown(
                            latest_tidy.to_html(index=False, float_format="{:.2f}".format),
                            unsafe_allow_html=True,
                        )
                else:
                    st.info(
                        "No se pudieron calcular correlaciones móviles con la ventana seleccionada."
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
            try:
                prices_df = tasvc.portfolio_history(
                    simbolos=portfolio_symbols, period="1y"
                )
            except AppError as err:
                st.error(str(err))
                return
            except Exception:
                logger.exception(
                    "Error al obtener históricos para análisis de riesgo",
                )
                st.error(
                    "No se pudieron obtener datos históricos, intente nuevamente más tarde",
                )
                return
        if prices_df.empty:
            st.info(
                "No se pudieron obtener datos históricos para calcular métricas de riesgo."
            )
        else:
            returns_df = compute_returns(prices_df)
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
                benchmark_labels = {
                    "S&P 500 (^GSPC)": "^GSPC",
                    "MERVAL": "MERVAL",
                    "Nasdaq (^IXIC)": "^IXIC",
                    "Dow Jones (^DJI)": "^DJI",
                }
                benchmark_choice = st.selectbox(
                    "Benchmark para beta y drawdown",
                    list(benchmark_labels.keys()),
                    index=0,
                    key="risk_benchmark_select",
                )
                benchmark_symbol = benchmark_labels[benchmark_choice]

                try:
                    bench_df = tasvc.portfolio_history(
                        simbolos=[benchmark_symbol], period="1y"
                    )
                except AppError as err:
                    st.error(str(err))
                    return
                except Exception:
                    logger.exception(
                        "Error al obtener benchmark para análisis de riesgo",
                    )
                    st.error(
                        "No se pudieron obtener datos históricos para el benchmark seleccionado.",
                    )
                    return

                bench_ret = compute_returns(bench_df).squeeze()
                if bench_ret.empty:
                    st.info(
                        "El benchmark seleccionado no tiene datos suficientes para calcular beta."
                    )
                    return

                confidence_options = {"90%": 0.90, "95%": 0.95, "99%": 0.99}
                selected_conf_label = st.selectbox(
                    "Nivel de confianza para VaR/CVaR",
                    list(confidence_options.keys()),
                    index=1,
                    key="var_confidence_select",
                )
                var_confidence = confidence_options[selected_conf_label]

                (
                    vol,
                    b,
                    var_value,
                    cvar_value,
                    opt_w,
                    port_ret,
                    asset_vols,
                    asset_drawdowns,
                    port_drawdown,
                ) = compute_risk_metrics(
                    returns_df,
                    bench_ret,
                    weights,
                    var_confidence=var_confidence,
                )

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric(
                    "Volatilidad anualizada",
                    f"{vol:.2%}" if vol == vol else "N/A",
                )
                c2.metric(
                    f"Beta vs {benchmark_choice}",
                    f"{b:.2f}" if b == b else "N/A",
                )
                tail_pct = (1 - var_confidence) * 100
                c3.metric(
                    f"VaR {tail_pct:.0f}%",
                    f"{var_value:.2%}" if var_value == var_value else "N/A",
                )
                c4.metric(
                    f"CVaR {tail_pct:.0f}%",
                    f"{cvar_value:.2%}" if cvar_value == cvar_value else "N/A",
                )
                c5.metric(
                    "Drawdown máximo",
                    f"{port_drawdown:.2%}" if port_drawdown == port_drawdown else "N/A",
                )

                vol_draw_cols = st.columns(2)
                if not asset_vols.empty:
                    vol_df = (
                        asset_vols.sort_values(ascending=False)
                        .rename("Volatilidad")
                        .reset_index()
                        .rename(columns={"index": "Símbolo"})
                    )
                    fig_vol_dist = px.bar(
                        vol_df,
                        x="Símbolo",
                        y="Volatilidad",
                        color="Símbolo",
                    )
                    fig_vol_dist = _apply_layout(
                        fig_vol_dist,
                        title="Distribución de volatilidades",
                        show_legend=False,
                    )
                    vol_draw_cols[0].plotly_chart(
                        fig_vol_dist,
                        width="stretch",
                        config=PLOTLY_CONFIG,
                    )
                else:
                    vol_draw_cols[0].info(
                        "No hay datos suficientes para calcular volatilidad por activo.",
                    )

                drawdown_series_port = drawdown_series(port_ret)
                if not drawdown_series_port.empty:
                    fig_drawdown = px.line(
                        drawdown_series_port,
                        labels={
                            "index": "Fecha",
                            "value": "Drawdown",
                        },
                    )
                    fig_drawdown = _apply_layout(
                        fig_drawdown,
                        title="Evolución del drawdown del portafolio",
                        show_legend=False,
                        y0_line=True,
                    )
                    vol_draw_cols[1].plotly_chart(
                        fig_drawdown,
                        width="stretch",
                        config=PLOTLY_CONFIG,
                    )
                else:
                    vol_draw_cols[1].info(
                        "No hay datos suficientes para calcular drawdown del portafolio.",
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
                        width="stretch",
                        config=PLOTLY_CONFIG,
                    )
                    st.caption(
                        "La volatilidad refleja la variabilidad de los retornos; aquí se muestra en una ventana móvil de 30 días."
                    )

                with st.expander("Distribución de retornos, VaR y CVaR"):
                    var_threshold = np.quantile(port_ret, 1 - var_confidence)
                    cvar_threshold = -cvar_value
                    fig_var = px.histogram(port_ret, nbins=50)
                    fig_var.add_vline(
                        x=var_threshold,
                        line_color="red",
                        annotation_text=f"VaR {tail_pct:.0f}%",
                        annotation_position="top left",
                    )
                    fig_var.add_vline(
                        x=cvar_threshold,
                        line_color="orange",
                        annotation_text=f"CVaR {tail_pct:.0f}%",
                        annotation_position="top right",
                    )
                    st.plotly_chart(
                        fig_var,
                        width="stretch",
                        config=PLOTLY_CONFIG,
                    )
                    st.caption(
                        (
                            "Las líneas indican los niveles de pérdida esperada: VaR (rojo) y CVaR (naranja)"
                            f" para un nivel de confianza del {var_confidence:.0%}."
                        )
                    )

                with st.expander("Beta vs retorno"):
                    scatter_cols = st.columns(1)
                    if returns_df.empty or bench_ret.empty:
                        scatter_cols[0].info(
                            "No hay datos suficientes para calcular beta por activo."
                        )
                    else:
                        avg_returns = returns_df.mean().fillna(0.0) * 252
                        betas = returns_df.apply(lambda s: beta(s, bench_ret), axis=0)
                        scatter_df = pd.DataFrame(
                            {
                                "Beta": betas,
                                "Retorno anualizado": avg_returns,
                            }
                        )
                        scatter_df["Símbolo"] = scatter_df.index
                        if favorites:
                            favorite_set = {favorites.normalize(sym) for sym in favorites.list()}
                        else:
                            favorite_set = set()
                        scatter_df["Favorito"] = scatter_df["Símbolo"].apply(
                            lambda sym: sym in favorite_set
                        )
                        scatter_df["Tamaño"] = np.where(
                            scatter_df["Favorito"],
                            18,
                            10,
                        )
                        scatter_df = scatter_df.replace([np.inf, -np.inf], np.nan).dropna()
                        if scatter_df.empty:
                            scatter_cols[0].info(
                                "No hay datos suficientes para graficar beta vs retorno.",
                            )
                        else:
                            fig_scatter = px.scatter(
                                scatter_df,
                                x="Beta",
                                y="Retorno anualizado",
                                size="Tamaño",
                                color="Favorito",
                                hover_name="Símbolo",
                                text="Símbolo",
                                color_discrete_map={
                                    True: "gold",
                                    False: "#636EFA",
                                },
                            )
                            fig_scatter.update_traces(marker=dict(line=dict(width=1, color="#2a2a2a")))
                            fig_scatter = _apply_layout(
                                fig_scatter,
                                title="Beta vs retorno anualizado",
                            )
                            scatter_cols[0].plotly_chart(
                                fig_scatter,
                                width="stretch",
                                config=PLOTLY_CONFIG,
                            )

                with st.expander("Optimización de portafolio (Markowitz)"):
                    opt_df = pd.DataFrame({"ticker": opt_w.index, "weight": opt_w.values})
                    st.bar_chart(opt_df, x="ticker", y="weight")
                    st.caption(
                        "Barras con la proporción que el modelo recomienda invertir en cada activo para equilibrar riesgo y retorno."
                    )

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
                    st.caption(
                        "Simula muchos escenarios posibles para estimar cómo podría variar el valor de la cartera en el futuro."
                    )

                with st.expander("Aplicar shocks"):
                    templates = {"Leve": 0.03, "Moderado": 0.07, "Fuerte": 0.12}
                    tmpl = st.selectbox("Escenario", list(templates), index=0)
                    shocks = {sym: -templates[tmpl] for sym in returns_df.columns}
                    st.caption(
                        f"Aplicando un shock uniforme de {templates[tmpl]:.0%} a todos los activos."
                    )
                base_prices = pd.Series(1.0, index=weights.index)
                stressed_val = apply_stress(base_prices, weights, shocks)
                st.write(f"Retorno con shocks: {stressed_val - 1:.2%}")
    else:
        st.info("No hay símbolos en el portafolio para analizar.")
