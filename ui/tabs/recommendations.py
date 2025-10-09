from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from application.portfolio_service import PortfolioService
from application.recommendation_service import RecommendationService
from application.screener.opportunities import run_screener_stub
from application.ta_service import TAService
from services.portfolio_view import compute_symbol_risk_metrics


LOGGER = logging.getLogger(__name__)
_FORM_KEY = "recommendations_form"
_INSIGHT_MESSAGES = {
    "diversify": "La selección prioriza un equilibrio sectorial con un riesgo promedio.",
    "max_return": "La estrategia busca maximizar retorno aceptando una volatilidad más elevada.",
    "low_risk": "La propuesta favorece estabilidad y preservación del capital frente a grandes oscilaciones.",
}


def _get_stored_state() -> dict:
    state = getattr(st, "session_state", {})
    stored = state.get(_SESSION_STATE_KEY)
    return stored if isinstance(stored, dict) else {}


def _expected_return_map(opportunities: pd.DataFrame) -> dict[str, float]:
    if not isinstance(opportunities, pd.DataFrame) or opportunities.empty:
        return {}
    frame = opportunities.copy()
    if "symbol" not in frame.columns and "ticker" in frame.columns:
        frame = frame.rename(columns={"ticker": "symbol"})
    frame["symbol"] = frame.get("symbol", pd.Series(dtype=str)).astype("string").fillna("").str.upper()
    try:
        expected = frame.apply(RecommendationService._expected_return, axis=1)
    except Exception:  # pragma: no cover - defensive
        LOGGER.debug("Fallo al calcular rentabilidad esperada desde oportunidades", exc_info=True)
        return {}
    expected = pd.to_numeric(expected, errors="coerce")
    return {
        str(sym): float(value)
        for sym, value in zip(frame["symbol"], expected)
        if str(sym) and np.isfinite(value)
    }


def _beta_lookup(
    risk_metrics: pd.DataFrame,
    opportunities: pd.DataFrame,
) -> dict[str, float]:
    lookup: dict[str, float] = {}
    if isinstance(risk_metrics, pd.DataFrame) and not risk_metrics.empty:
        df = risk_metrics.copy()
        symbol_col = "simbolo" if "simbolo" in df.columns else "symbol"
        df[symbol_col] = df.get(symbol_col, pd.Series(dtype=str)).astype("string").fillna("").str.upper()
        betas = pd.to_numeric(df.get("beta"), errors="coerce")
        for sym, beta_val in zip(df[symbol_col], betas):
            if not sym or not np.isfinite(beta_val):
                continue
            lookup[str(sym)] = float(beta_val)

    if isinstance(opportunities, pd.DataFrame) and not opportunities.empty:
        frame = opportunities.copy()
        if "symbol" not in frame.columns and "ticker" in frame.columns:
            frame = frame.rename(columns={"ticker": "symbol"})
        frame["symbol"] = frame.get("symbol", pd.Series(dtype=str)).astype("string").fillna("").str.upper()
        sectors = frame.get("sector", pd.Series(dtype=str))
        for sym, sector in zip(frame["symbol"], sectors):
            symbol = str(sym)
            if not symbol or symbol in lookup:
                continue
            lookup[symbol] = RecommendationService._estimate_beta_from_sector(str(sector or ""))
    return lookup


def _mean_numeric(series: pd.Series | None) -> float:
    if series is None:
        return float("nan")
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return float("nan")
    return float(numeric.mean())


def _render_automatic_insight(
    recommendations: pd.DataFrame,
    *,
    mode_key: str,
    beta_lookup: dict[str, float] | None = None,
) -> None:
    if recommendations.empty:
        return

    expected_mean = _mean_numeric(recommendations.get("expected_return"))

    beta_series = None
    if "beta" in recommendations.columns:
        beta_series = recommendations["beta"]
    elif beta_lookup:
        symbols = recommendations.get("symbol", pd.Series(dtype=str)).astype("string")
        betas = [beta_lookup.get(str(symbol).upper()) for symbol in symbols]
        beta_series = pd.Series(betas)
    beta_mean = _mean_numeric(beta_series)

    mode_message = _INSIGHT_MESSAGES.get(mode_key, _INSIGHT_MESSAGES["diversify"])

    lines = ["**Insight automático**", mode_message]
    metrics_parts: list[str] = []
    if np.isfinite(expected_mean):
        metrics_parts.append(f"Rentabilidad esperada promedio: {_format_percent(expected_mean)}")
    if np.isfinite(beta_mean):
        metrics_parts.append(f"Beta promedio: {_format_float(beta_mean)}")
    if metrics_parts:
        lines.append(" | ".join(metrics_parts))

    st.info("\n\n".join(lines))


def _format_currency(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"${value:,.0f}".replace(",", ".")


def _format_percent(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.2f}%"


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.2f}"


def _format_currency_delta(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-6:
        return "-"
    sign = "+" if value > 0 else "-"
    return f"{sign}${abs(value):,.0f}".replace(",", ".")


def _format_percent_delta(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-6:
        return "-"
    sign = "+" if value > 0 else "-"
    return f"{sign}{abs(value):.2f}%"


def _format_float_delta(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-6:
        return "-"
    sign = "+" if value > 0 else "-"
    return f"{sign}{abs(value):.2f}"


def _render_simulation_results(result: dict[str, dict[str, float]]) -> None:
    if not isinstance(result, dict) or not result:
        return

    before = result.get("before") or {}
    after = result.get("after") or {}

    def _to_float(value: float | str | None, default: float) -> float:
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
        return parsed

    before_value = _to_float(before.get("total_value"), 0.0)
    after_value = _to_float(after.get("total_value"), before_value)
    before_return = _to_float(before.get("projected_return"), 0.0)
    after_return = _to_float(after.get("projected_return"), before_return)
    before_beta = _to_float(before.get("beta"), np.nan)
    after_beta = _to_float(after.get("beta"), before_beta)
    additional = _to_float(after.get("additional_investment"), after_value - before_value)

    rows = [
        {
            "Métrica": "Valorizado total",
            "Antes": _format_currency(before_value),
            "Después": _format_currency(after_value),
            "Variación": _format_currency_delta(after_value - before_value),
        },
        {
            "Métrica": "Rentabilidad proyectada",
            "Antes": _format_percent(before_return),
            "Después": _format_percent(after_return),
            "Variación": _format_percent_delta(after_return - before_return),
        },
        {
            "Métrica": "Beta / Riesgo total",
            "Antes": _format_float(before_beta),
            "Después": _format_float(after_beta),
            "Variación": _format_float_delta(after_beta - before_beta),
        },
    ]

    st.markdown("#### Simulación de impacto (Antes vs. Después)")
    st.table(pd.DataFrame(rows))

    if np.isfinite(additional) and additional > 0:
        st.caption(
            "Se asignan "
            f"{_format_currency(additional)} adicionales siguiendo la distribución sugerida."
        )
_SESSION_STATE_KEY = "_recommendations_state"


def _get_stored_state() -> dict:
    state = getattr(st, "session_state", {})
    stored = state.get(_SESSION_STATE_KEY)
    return stored if isinstance(stored, dict) else {}


def _expected_return_map(opportunities: pd.DataFrame) -> dict[str, float]:
    if not isinstance(opportunities, pd.DataFrame) or opportunities.empty:
        return {}
    frame = opportunities.copy()
    if "symbol" not in frame.columns and "ticker" in frame.columns:
        frame = frame.rename(columns={"ticker": "symbol"})
    frame["symbol"] = frame.get("symbol", pd.Series(dtype=str)).astype("string").fillna("").str.upper()
    try:
        expected = frame.apply(RecommendationService._expected_return, axis=1)
    except Exception:  # pragma: no cover - defensive
        LOGGER.debug("Fallo al calcular rentabilidad esperada desde oportunidades", exc_info=True)
        return {}
    expected = pd.to_numeric(expected, errors="coerce")
    return {
        str(sym): float(value)
        for sym, value in zip(frame["symbol"], expected)
        if str(sym) and np.isfinite(value)
    }


def _beta_lookup(
    risk_metrics: pd.DataFrame,
    opportunities: pd.DataFrame,
) -> dict[str, float]:
    lookup: dict[str, float] = {}
    if isinstance(risk_metrics, pd.DataFrame) and not risk_metrics.empty:
        df = risk_metrics.copy()
        symbol_col = "simbolo" if "simbolo" in df.columns else "symbol"
        df[symbol_col] = df.get(symbol_col, pd.Series(dtype=str)).astype("string").fillna("").str.upper()
        betas = pd.to_numeric(df.get("beta"), errors="coerce")
        for sym, beta_val in zip(df[symbol_col], betas):
            if not sym or not np.isfinite(beta_val):
                continue
            lookup[str(sym)] = float(beta_val)

    if isinstance(opportunities, pd.DataFrame) and not opportunities.empty:
        frame = opportunities.copy()
        if "symbol" not in frame.columns and "ticker" in frame.columns:
            frame = frame.rename(columns={"ticker": "symbol"})
        frame["symbol"] = frame.get("symbol", pd.Series(dtype=str)).astype("string").fillna("").str.upper()
        sectors = frame.get("sector", pd.Series(dtype=str))
        for sym, sector in zip(frame["symbol"], sectors):
            symbol = str(sym)
            if not symbol or symbol in lookup:
                continue
            lookup[symbol] = RecommendationService._estimate_beta_from_sector(str(sector or ""))
    return lookup


def _format_currency(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"${value:,.0f}".replace(",", ".")


def _format_percent(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.2f}%"


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.2f}"


def _format_currency_delta(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-6:
        return "-"
    sign = "+" if value > 0 else "-"
    return f"{sign}${abs(value):,.0f}".replace(",", ".")


def _format_percent_delta(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-6:
        return "-"
    sign = "+" if value > 0 else "-"
    return f"{sign}{abs(value):.2f}%"


def _format_float_delta(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-6:
        return "-"
    sign = "+" if value > 0 else "-"
    return f"{sign}{abs(value):.2f}"


def _render_simulation_results(result: dict[str, dict[str, float]]) -> None:
    if not isinstance(result, dict) or not result:
        return

    before = result.get("before") or {}
    after = result.get("after") or {}

    def _to_float(value: float | str | None, default: float) -> float:
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
        return parsed

    before_value = _to_float(before.get("total_value"), 0.0)
    after_value = _to_float(after.get("total_value"), before_value)
    before_return = _to_float(before.get("projected_return"), 0.0)
    after_return = _to_float(after.get("projected_return"), before_return)
    before_beta = _to_float(before.get("beta"), np.nan)
    after_beta = _to_float(after.get("beta"), before_beta)
    additional = _to_float(after.get("additional_investment"), after_value - before_value)

    rows = [
        {
            "Métrica": "Valorizado total",
            "Antes": _format_currency(before_value),
            "Después": _format_currency(after_value),
            "Variación": _format_currency_delta(after_value - before_value),
        },
        {
            "Métrica": "Rentabilidad proyectada",
            "Antes": _format_percent(before_return),
            "Después": _format_percent(after_return),
            "Variación": _format_percent_delta(after_return - before_return),
        },
        {
            "Métrica": "Beta / Riesgo total",
            "Antes": _format_float(before_beta),
            "Después": _format_float(after_beta),
            "Variación": _format_float_delta(after_beta - before_beta),
        },
    ]

    st.markdown("#### Simulación de impacto (Antes vs. Después)")
    st.table(pd.DataFrame(rows))

    if np.isfinite(additional) and additional > 0:
        st.caption(
            "Se asignan "
            f"{_format_currency(additional)} adicionales siguiendo la distribución sugerida."
        )
_SESSION_STATE_KEY = "_recommendations_state"


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


def _render_analysis_summary(source: RecommendationService | dict[str, object]) -> None:
    if isinstance(source, RecommendationService):
        analysis = source.analyze_portfolio()
    elif isinstance(source, dict):
        analysis = source
    else:
        return
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


def _render_recommendations_visuals(
    result: pd.DataFrame, *, mode_label: str, amount: float
) -> None:
    if result.empty:
        return

    amount_text = f"${amount:,.0f}".replace(",", ".")
    st.markdown(
        f"**Modo seleccionado:** {mode_label} &nbsp;|&nbsp; "
        f"**Monto a invertir:** {amount_text}"
    )

    pie_fig = px.pie(
        result,
        names="symbol",
        values="allocation_%",
        hole=0.35,
    )
    pie_fig.update_traces(textposition="inside", texttemplate="%{label}<br>%{value:.2f}%")
    pie_fig.update_layout(
        margin=dict(t=40, b=0, l=0, r=0),
        title="Distribución sugerida por activo",
    )

    bar_data = result.sort_values("allocation_amount", ascending=False)
    bar_fig = px.bar(
        bar_data,
        x="symbol",
        y="allocation_amount",
        text="allocation_amount",
    )
    bar_fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside", cliponaxis=False)
    bar_fig.update_layout(
        margin=dict(t=40, b=40, l=0, r=0),
        title="Montos asignados (ARS)",
        yaxis_title="Monto",
        xaxis_title="Símbolo",
    )

    charts = st.columns(2)
    with charts[0]:
        st.plotly_chart(pie_fig, use_container_width=True, config={"displayModeBar": False})
    with charts[1]:
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})


def render_recommendations_tab() -> None:
    """Render the recommendations tab inside the main dashboard."""

    positions = _get_portfolio_positions()
    st.header("🔮 Recomendaciones personalizadas")
    st.caption(
        "Proyectá cómo complementar tu cartera con nuevas ideas balanceadas según tu perfil."
    )

    if positions.empty:
        try:
            st.session_state.pop(_SESSION_STATE_KEY, None)
        except Exception:  # pragma: no cover - defensive safeguard
            LOGGER.debug("No se pudo limpiar el estado de recomendaciones", exc_info=True)
        st.info(
            "Necesitamos un portafolio cargado para sugerir ideas. Revisá la pestaña "
            "Portafolio y actualizá tus posiciones."
        )
        return

    stored_state = _get_stored_state()
    recommendations = pd.DataFrame()
    opportunities = pd.DataFrame()
    risk_metrics = pd.DataFrame()
    stored_amount: float | None = None
    stored_mode_label: str | None = None
    stored_mode_key: str | None = None
    analysis_data: dict[str, object] | None = None

    if stored_state:
        rec_df = stored_state.get("recommendations")
        if isinstance(rec_df, pd.DataFrame):
            recommendations = rec_df.copy()
        opp_df = stored_state.get("opportunities")
        if isinstance(opp_df, pd.DataFrame):
            opportunities = opp_df.copy()
        risk_df = stored_state.get("risk_metrics")
        if isinstance(risk_df, pd.DataFrame):
            risk_metrics = risk_df.copy()
        try:
            stored_amount_val = stored_state.get("amount")
            if stored_amount_val is not None:
                stored_amount = float(stored_amount_val)
        except (TypeError, ValueError):
            stored_amount = None
        mode_candidate = stored_state.get("mode_label")
        if isinstance(mode_candidate, str) and mode_candidate:
            stored_mode_label = mode_candidate
        mode_key_candidate = stored_state.get("mode_key")
        if isinstance(mode_key_candidate, str) and mode_key_candidate:
            stored_mode_key = mode_key_candidate
        analysis_candidate = stored_state.get("analysis")
        if isinstance(analysis_candidate, dict):
            analysis_data = analysis_candidate

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

    if submitted:
        with st.spinner("Analizando portafolio y oportunidades..."):
            fundamentals = _load_portfolio_fundamentals(symbols)
            risk_metrics = _load_risk_metrics(symbols)
            if not isinstance(risk_metrics, pd.DataFrame):
                risk_metrics = pd.DataFrame()
            screener_result = run_screener_stub(include_technicals=False)
            if isinstance(screener_result, tuple):
                opportunities = screener_result[0]
            else:
                opportunities = screener_result
            if not isinstance(opportunities, pd.DataFrame):
                opportunities = pd.DataFrame()
            svc = RecommendationService(
                portfolio_df=positions,
                opportunities_df=opportunities,
                risk_metrics_df=risk_metrics,
                fundamentals_df=fundamentals,
            )
            recommendations = svc.recommend(amount, mode=mode[0])
            analysis_data = svc.analyze_portfolio()

            if recommendations.empty:
                st.session_state.pop(_SESSION_STATE_KEY, None)
            else:
                payload = {
                    "recommendations": recommendations.copy(),
                    "opportunities": opportunities.copy(),
                    "risk_metrics": risk_metrics.copy(),
                    "amount": amount,
                    "mode_label": mode[1],
                    "mode_key": mode[0],
                    "analysis": analysis_data,
                }
                try:
                    st.session_state[_SESSION_STATE_KEY] = payload
                except Exception:  # pragma: no cover - defensive safeguard
                    LOGGER.debug(
                        "No se pudo guardar el estado de recomendaciones", exc_info=True
                    )
            stored_amount = amount
            stored_mode_label = mode[1]

    if isinstance(analysis_data, dict):
        _render_analysis_summary(analysis_data)
    elif isinstance(stored_state.get("analysis"), dict):
        _render_analysis_summary(stored_state["analysis"])

    amount_to_display = amount
    if stored_amount is not None and np.isfinite(stored_amount):
        amount_to_display = stored_amount
    mode_label_to_display = stored_mode_label or mode[1]
    mode_key_to_display = stored_mode_key or mode[0]

    expected_map = _expected_return_map(opportunities)
    beta_lookup = _beta_lookup(risk_metrics, opportunities)

    if not recommendations.empty:
        _render_recommendations_visuals(
            recommendations,
            mode_label=mode_label_to_display,
            amount=amount_to_display,
        )
    _render_recommendations_table(recommendations)
    _render_automatic_insight(
        recommendations,
        mode_key=mode_key_to_display,
        beta_lookup=beta_lookup,
    )

    simulate_clicked = st.button(
        "Simular impacto",
        disabled=recommendations.empty,
    )

    if simulate_clicked:
        session = getattr(st, "session_state", {})
        totals = session.get("portfolio_last_totals")
        portfolio_service = PortfolioService()
        try:
            result = portfolio_service.simulate_allocation(
                portfolio_positions=positions,
                totals=totals,
                recommendations=recommendations,
                expected_returns=expected_map,
                betas=beta_lookup,
            )
        except Exception:
            LOGGER.exception("No se pudo simular el impacto de las recomendaciones")
            st.error("No se pudo simular el impacto. Intentá nuevamente más tarde.")
        else:
            _render_simulation_results(result)

    simulate_clicked = st.button(
        "Simular impacto",
        disabled=recommendations.empty,
    )

    if simulate_clicked:
        session = getattr(st, "session_state", {})
        totals = session.get("portfolio_last_totals")
        expected_map = _expected_return_map(opportunities)
        beta_map = _beta_lookup(risk_metrics, opportunities)
        portfolio_service = PortfolioService()
        try:
            result = portfolio_service.simulate_allocation(
                portfolio_positions=positions,
                totals=totals,
                recommendations=recommendations,
                expected_returns=expected_map,
                betas=beta_map,
            )
        except Exception:
            LOGGER.exception("No se pudo simular el impacto de las recomendaciones")
            st.error("No se pudo simular el impacto. Intentá nuevamente más tarde.")
        else:
            _render_simulation_results(result)


__all__ = ["render_recommendations_tab"]

