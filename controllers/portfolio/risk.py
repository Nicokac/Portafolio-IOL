import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from application.risk_service import (
    compute_returns,
    annualized_volatility,
    beta,
    historical_var,
    markowitz_optimize,
    monte_carlo_simulation,
    apply_stress,
)
from ui.charts import plot_correlation_heatmap
from ui.export import PLOTLY_CONFIG


def compute_risk_metrics(returns_df, bench_ret, weights):
    """Compute core risk metrics for the portfolio."""
    port_ret = returns_df.mul(weights, axis=1).sum(axis=1)
    vol = annualized_volatility(port_ret)
    b = beta(port_ret, bench_ret)
    var_95 = historical_var(port_ret)
    opt_w = markowitz_optimize(returns_df)
    return vol, b, var_95, opt_w, port_ret


def render_risk_analysis(df_view, tasvc):
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
                vol, b, var_95, opt_w, port_ret = compute_risk_metrics(
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
                        width="stretch",
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
                        width="stretch",
                        config=PLOTLY_CONFIG,
                    )
                    st.caption(
                        "La línea roja indica el VaR al 5%, representando la pérdida máxima esperada con 95% de confianza."
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
