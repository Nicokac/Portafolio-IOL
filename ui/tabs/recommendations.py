from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from application.recommendation_service import RecommendationService
from application.screener.opportunities import run_screener_stub
from application.ta_service import TAService
from services.portfolio_view import compute_symbol_risk_metrics


LOGGER = logging.getLogger(__name__)
_FORM_KEY = "recommendations_form"


def _get_portfolio_positions() -> pd.DataFrame:
    session = getattr(st, "session_state", {})
    df = session.get("portfolio_last_positions")
    if isinstance(df, pd.DataFrame):
        return df.copy()
    viewmodel = session.get("portfolio_last_viewmodel")
    if getattr(viewmodel, "positions", None) is not None:
        positions = getattr(viewmodel, "positions")
        if isinstance(positions, pd.DataFrame):
            return positions.copy()
    return pd.DataFrame()


def _load_portfolio_fundamentals(symbols: list[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    tasvc = TAService()
    try:
        fundamentals = tasvc.portfolio_fundamentals(symbols)
    except Exception:
        LOGGER.exception("No se pudieron obtener fundamentals para recomendaciones")
        return pd.DataFrame()
    return fundamentals


def _load_risk_metrics(symbols: list[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    tasvc = TAService()
    try:
        return compute_symbol_risk_metrics(
            tasvc,
            symbols,
            benchmark="^GSPC",
            period="6mo",
        )
    except Exception:
        LOGGER.exception("No se pudieron obtener métricas de riesgo para recomendaciones")
        return pd.DataFrame()


def _render_analysis_summary(service: RecommendationService) -> None:
    analysis = service.analyze_portfolio()
    cols = st.columns(3)
    with cols[0]:
        types = analysis.get("type_distribution")
        if isinstance(types, pd.Series) and not types.empty:
            st.metric(
                "Tipo dominante",
                f"{types.index[0]} ({types.iloc[0] * 100:.1f}%)",
            )
    with cols[1]:
        sectors = analysis.get("sector_distribution")
        if isinstance(sectors, pd.Series) and not sectors.empty:
            st.metric(
                "Sector principal",
                f"{sectors.index[0]} ({sectors.iloc[0] * 100:.1f}%)",
            )
    with cols[2]:
        currency = analysis.get("currency_distribution")
        if isinstance(currency, pd.Series) and not currency.empty:
            st.metric(
                "Moneda predominante",
                f"{currency.index[0]} ({currency.iloc[0] * 100:.1f}%)",
            )

    beta_dist = analysis.get("beta_distribution")
    if isinstance(beta_dist, pd.Series) and not beta_dist.empty:
        beta_summary = ", ".join(
            f"{bucket}: {share * 100:.1f}%" for bucket, share in beta_dist.items()
        )
        st.caption(f"Distribución de beta: {beta_summary}")


def _render_recommendations_table(result: pd.DataFrame) -> None:
    if result.empty:
        st.info("Ingresá un monto para calcular sugerencias personalizadas.")
        return

    formatted = result.copy()
    formatted["allocation_%"] = formatted["allocation_%"].map(lambda v: f"{v:.2f}%")
    formatted["allocation_amount"] = formatted["allocation_amount"].map(
        lambda v: f"${v:,.0f}".replace(",", ".")
    )
    st.dataframe(
        formatted,
        use_container_width=True,
        hide_index=True,
    )


def render_recommendations_tab() -> None:
    """Render the recommendations tab inside the main dashboard."""

    positions = _get_portfolio_positions()
    st.header("🔮 Recomendaciones personalizadas")
    st.caption(
        "Proyectá cómo complementar tu cartera con nuevas ideas balanceadas según tu perfil."
    )

    if positions.empty:
        st.info(
            "Necesitamos un portafolio cargado para sugerir ideas. Revisá la pestaña "
            "Portafolio y actualizá tus posiciones."
        )
        return

    symbols = [str(sym) for sym in positions.get("simbolo", []) if sym]

    with st.form(_FORM_KEY):
        amount = st.number_input(
            "Monto disponible a invertir (ARS)",
            min_value=0.0,
            value=100_000.0,
            step=10_000.0,
            format="%0.0f",
        )
        mode = st.selectbox(
            "Modo de recomendación",
            options=[
                ("diversify", "Diversificar"),
                ("max_return", "Maximizar retorno"),
                ("low_risk", "Bajar riesgo"),
            ],
            format_func=lambda item: item[1],
            index=0,
        )
        submitted = st.form_submit_button("Calcular recomendaciones")

    recommendations = pd.DataFrame()
    if submitted:
        with st.spinner("Analizando portafolio y oportunidades..."):
            fundamentals = _load_portfolio_fundamentals(symbols)
            risk_metrics = _load_risk_metrics(symbols)
            screener_result = run_screener_stub(include_technicals=False)
            if isinstance(screener_result, tuple):
                opportunities = screener_result[0]
            else:
                opportunities = screener_result
            svc = RecommendationService(
                portfolio_df=positions,
                opportunities_df=opportunities,
                risk_metrics_df=risk_metrics,
                fundamentals_df=fundamentals,
            )
            recommendations = svc.recommend(amount, mode=mode[0])
            _render_analysis_summary(svc)

    _render_recommendations_table(recommendations)


__all__ = ["render_recommendations_tab"]

