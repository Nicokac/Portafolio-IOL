from __future__ import annotations

import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from application.benchmark_service import (
    BENCHMARK_BASELINES,
    compute_benchmark_comparison,
)
from application.portfolio_service import PortfolioService
from application.profile_service import DEFAULT_PROFILE, ProfileService
from application.recommendation_service import RecommendationService
from application.screener.opportunities import run_screener_stub
from application.ta_service import TAService
from application.predictive_service import get_cache_stats, predict_sector_performance
from services.portfolio_view import compute_symbol_risk_metrics


LOGGER = logging.getLogger(__name__)


def _suppress_streamlit_bare_mode_warnings() -> None:
    """Reduce noisy bare-mode warnings when rendering without Streamlit runner."""

    logger_names = (
        "streamlit.runtime.scriptrunner_utils.script_run_context",
        "streamlit.runtime.state.session_state",
        "streamlit.runtime.scriptrunner_utils",  # fallback for future versions
        "streamlit",
    )
    for name in logger_names:
        try:
            logging.getLogger(name).setLevel(logging.ERROR)
            logging.getLogger(name).propagate = False
        except Exception:  # pragma: no cover - defensive guard for logging internals
            LOGGER.debug("No se pudo ajustar el nivel de logging para %s", name, exc_info=True)
_FORM_KEY = "recommendations_form"
_SESSION_STATE_KEY = "_recommendations_state"
_MODE_OPTIONS = [
    ("diversify", "Diversificar"),
    ("max_return", "Maximizar retorno"),
    ("low_risk", "Bajar riesgo"),
]
_MODE_LABELS = {key: label for key, label in _MODE_OPTIONS}
_MODE_ALIASES = {
    "diversificar": "diversify",
    "diversify": "diversify",
    "max_return": "max_return",
    "maximizar retorno": "max_return",
    "maximizar": "max_return",
    "low_risk": "low_risk",
    "bajar riesgo": "low_risk",
    "low risk": "low_risk",
}
_INSIGHT_MESSAGES = {
    "diversify": "La selección prioriza un equilibrio sectorial con un riesgo promedio.",
    "max_return": "La estrategia busca maximizar retorno aceptando una volatilidad más elevada.",
    "low_risk": "La propuesta favorece estabilidad y preservación del capital frente a grandes oscilaciones.",
}

_PROFILE_RISK_OPTIONS = [
    ("bajo", "Conservador"),
    ("medio", "Moderado"),
    ("alto", "Dinámico"),
]
_PROFILE_HORIZON_OPTIONS = [
    ("corto", "3 meses"),
    ("mediano", "12 meses"),
    ("largo", "24 meses o más"),
]
_BENCHMARK_LABELS = {
    key: value.get("name", key.upper()) for key, value in BENCHMARK_BASELINES.items()
}


def _resolve_mode(value: object) -> tuple[str, str]:
    if isinstance(value, tuple) and value:
        key = str(value[0])
        normalized_key = key.lower()
        resolved_key = _MODE_ALIASES.get(normalized_key, normalized_key)
        label = _MODE_LABELS.get(resolved_key, _MODE_LABELS["diversify"])
        return resolved_key, label
    if isinstance(value, str):
        normalized_value = value.strip().lower()
        resolved_key = _MODE_ALIASES.get(normalized_value, normalized_value)
        if resolved_key in _MODE_LABELS:
            return resolved_key, _MODE_LABELS[resolved_key]
    return "diversify", _MODE_LABELS["diversify"]


def _option_index(options: list[tuple[str, str]], value: str) -> int:
    for idx, item in enumerate(options):
        if item[0] == value:
            return idx
    return 0


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


def _render_profile_panel(service: ProfileService) -> dict[str, str]:
    profile = service.get_profile()
    st.markdown(f"**{service.badge_label(profile)}**")
    with st.expander("Configurar perfil inversor", expanded=False):
        st.caption(
            "Guardamos tus preferencias en el dispositivo para ajustar futuras recomendaciones."
        )
        risk_choice = st.selectbox(
            "Tolerancia al riesgo",
            options=_PROFILE_RISK_OPTIONS,
            format_func=lambda item: item[1],
            index=_option_index(_PROFILE_RISK_OPTIONS, profile["risk_tolerance"]),
            key=f"{_FORM_KEY}_profile_risk",
        )
        horizon_choice = st.selectbox(
            "Horizonte de inversión",
            options=_PROFILE_HORIZON_OPTIONS,
            format_func=lambda item: item[1],
            index=_option_index(_PROFILE_HORIZON_OPTIONS, profile["investment_horizon"]),
            key=f"{_FORM_KEY}_profile_horizon",
        )
        preferred_choice = st.selectbox(
            "Enfoque preferido",
            options=_MODE_OPTIONS,
            format_func=lambda item: item[1],
            index=_option_index(_MODE_OPTIONS, profile["preferred_mode"]),
            key=f"{_FORM_KEY}_profile_mode",
        )
    risk_value = risk_choice[0] if isinstance(risk_choice, tuple) else str(risk_choice)
    horizon_value = (
        horizon_choice[0] if isinstance(horizon_choice, tuple) else str(horizon_choice)
    )
    mode_value = (
        preferred_choice[0] if isinstance(preferred_choice, tuple) else str(preferred_choice)
    )
    updated = service.update_profile(
        risk_tolerance=risk_value,
        investment_horizon=horizon_value,
        preferred_mode=mode_value,
    )
    return updated


def _mean_numeric(series: pd.Series | None) -> float:
    if series is None:
        return float("nan")
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return float("nan")
    return float(numeric.mean())


def _weighted_mean(
    values: pd.Series | None, weights: pd.Series | None
) -> float:
    if values is None:
        return float("nan")
    numeric = pd.to_numeric(values, errors="coerce")
    if weights is None:
        return _mean_numeric(numeric)
    weight_series = pd.to_numeric(weights, errors="coerce")
    mask = np.isfinite(numeric) & np.isfinite(weight_series)
    numeric = numeric[mask]
    weight_series = weight_series[mask]
    if numeric.empty or weight_series.empty:
        return _mean_numeric(numeric)
    total_weight = float(weight_series.sum())
    if total_weight <= 0:
        return _mean_numeric(numeric)
    return float(np.average(numeric, weights=weight_series))


def _render_automatic_insight(
    recommendations: pd.DataFrame,
    *,
    mode_key: str,
    beta_lookup: dict[str, float] | None = None,
) -> None:
    if recommendations.empty:
        return

    weight_series = None
    if "allocation_%" in recommendations.columns:
        weight_series = pd.to_numeric(
            recommendations.get("allocation_%"), errors="coerce"
        )

    expected_mean = _weighted_mean(
        recommendations.get("expected_return"), weight_series
    )
    predicted_mean = _weighted_mean(
        recommendations.get("predicted_return_pct"), weight_series
    )

    beta_series = None
    if "beta" in recommendations.columns:
        beta_series = recommendations["beta"]
    elif beta_lookup:
        symbols = recommendations.get("symbol", pd.Series(dtype=str)).astype("string")
        betas = [beta_lookup.get(str(symbol).upper()) for symbol in symbols]
        beta_series = pd.Series(betas)
    beta_mean = _weighted_mean(beta_series, weight_series)

    mode_message = _INSIGHT_MESSAGES.get(mode_key, _INSIGHT_MESSAGES["diversify"])

    lines = ["**Insight automático**", mode_message]
    metrics_parts: list[str] = []
    if np.isfinite(expected_mean):
        metrics_parts.append(f"Rentabilidad esperada promedio: {_format_percent(expected_mean)}")
    if np.isfinite(predicted_mean):
        metrics_parts.append(
            f"Predicción sectorial ponderada: {_format_percent(predicted_mean)}"
        )
    if np.isfinite(beta_mean):
        metrics_parts.append(f"Beta promedio: {_format_float(beta_mean)}")
    sector_series = recommendations.get("sector")
    if isinstance(sector_series, pd.Series) and not sector_series.empty:
        normalized = sector_series.astype("string").str.strip()
        normalized = normalized[normalized != ""]
        if not normalized.empty:
            dominant_sector = normalized.value_counts().idxmax()
            metrics_parts.append(f"Sector dominante: {dominant_sector}")
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


def _build_numeric_lookup(df: pd.DataFrame, column: str) -> dict[str, float]:
    if df.empty or column not in df.columns:
        return {}
    symbols = (
        df.get("symbol", pd.Series(dtype=str))
        .astype("string")
        .fillna("")
        .str.upper()
    )
    values = pd.to_numeric(df[column], errors="coerce")
    return {
        str(symbol): float(value)
        for symbol, value in zip(symbols, values)
        if symbol and np.isfinite(value)
    }


def _enrich_recommendations(
    recommendations: pd.DataFrame,
    *,
    expected_returns: Mapping[str, float] | None,
    betas: Mapping[str, float] | None,
) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations

    enriched = recommendations.copy()
    symbols = (
        enriched.get("symbol", pd.Series(dtype=str))
        .astype("string")
        .fillna("")
        .str.upper()
    )

    expected_lookup: dict[str, float] = {}
    for key, value in (expected_returns or {}).items():
        symbol = str(key).upper()
        if not symbol:
            continue
        try:
            rate = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(rate):
            expected_lookup[symbol] = rate

    beta_lookup: dict[str, float] = {}
    for key, value in (betas or {}).items():
        symbol = str(key).upper()
        if not symbol:
            continue
        try:
            beta_val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(beta_val):
            beta_lookup[symbol] = beta_val

    expected_series = pd.Series(
        [expected_lookup.get(symbol, np.nan) for symbol in symbols], index=enriched.index
    )
    enriched["expected_return"] = pd.to_numeric(
        enriched.get("expected_return", expected_series), errors="coerce"
    )
    enriched["expected_return"] = enriched["expected_return"].fillna(expected_series)

    beta_series = pd.Series(
        [beta_lookup.get(symbol, np.nan) for symbol in symbols], index=enriched.index
    )
    enriched["beta"] = pd.to_numeric(enriched.get("beta", beta_series), errors="coerce")
    enriched["beta"] = enriched["beta"].fillna(beta_series)

    return enriched


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


def _render_benchmark_block(recommendations: pd.DataFrame) -> None:
    if recommendations.empty:
        return
    st.markdown("#### Comparativa con benchmarks")
    keys = list(_BENCHMARK_LABELS.keys()) or ["merval", "sp500", "bonos"]
    columns = st.columns(len(keys))
    for col, key in zip(columns, keys):
        metrics = compute_benchmark_comparison(recommendations, key)
        if not metrics:
            continue
        label = str(metrics.get("label") or _BENCHMARK_LABELS.get(key, key.upper()))
        delta_return = _format_percent_delta(float(metrics.get("relative_return", float("nan"))))
        delta_beta = _format_float_delta(float(metrics.get("relative_beta", float("nan"))))
        tracking = _format_percent(float(metrics.get("tracking_error", float("nan"))))
        portfolio_ret = _format_percent(float(metrics.get("portfolio_return", float("nan"))))
        benchmark_ret = _format_percent(float(metrics.get("benchmark_return", float("nan"))))
        with col:
            st.markdown(f"**{label}**")
            st.caption(
                f"ΔRetorno: {delta_return} | ΔBeta: {delta_beta} | Tracking Error: {tracking}"
            )
            st.caption(
                f"Portafolio: {portfolio_ret} &nbsp;/&nbsp; Índice: {benchmark_ret}"
            )


def _render_recommendations_table(
    result: pd.DataFrame, *, show_predictions: bool
) -> None:
    if result.empty:
        st.info("Ingresá un monto para calcular sugerencias personalizadas.")
        return

    formatted = result.copy()
    if "expected_return" in formatted.columns:
        formatted = formatted.drop(columns=["expected_return"])
    predicted_formatted = None
    if "predicted_return_pct" in formatted.columns:
        predicted_values = pd.to_numeric(
            formatted["predicted_return_pct"], errors="coerce"
        )
        if show_predictions:
            predicted_formatted = predicted_values.map(
                lambda v: f"{v:.2f}%" if np.isfinite(v) else "-"
            )
        formatted = formatted.drop(columns=["predicted_return_pct"])
    if "beta" in formatted.columns:
        formatted = formatted.drop(columns=["beta"])
    formatted["allocation_%"] = formatted["allocation_%"].map(lambda v: f"{v:.2f}%")
    formatted["allocation_amount"] = formatted["allocation_amount"].map(
        lambda v: f"${v:,.0f}".replace(",", ".")
    )
    if predicted_formatted is not None:
        formatted.insert(
            3,
            "Predicted Return (%)",
            predicted_formatted,
        )
    if "rationale_extended" in formatted.columns:
        formatted = formatted.rename(columns={"rationale_extended": "Racional extendido"})
    formatted = formatted.rename(columns={"rationale": "Racional"})
    st.dataframe(
        formatted,
        width="stretch",
        hide_index=True,
    )

    export_columns = [
        "symbol",
        "allocation_%",
        "allocation_amount",
        "expected_return",
        "predicted_return_pct",
        "beta",
        "rationale",
    ]
    export_df = result.copy()
    for column in export_columns:
        if column not in export_df.columns:
            export_df[column] = np.nan
    export_df = export_df[export_columns]

    summary = {col: "" for col in export_columns}
    summary["symbol"] = "Promedios"
    numeric_return = pd.to_numeric(export_df["expected_return"], errors="coerce")
    numeric_beta = pd.to_numeric(export_df["beta"], errors="coerce")
    expected_mean = numeric_return.dropna().mean()
    beta_mean = numeric_beta.dropna().mean()
    summary["expected_return"] = (
        round(float(expected_mean), 4) if pd.notna(expected_mean) else ""
    )
    summary["beta"] = round(float(beta_mean), 4) if pd.notna(beta_mean) else ""
    export_with_summary = pd.concat(
        [export_df, pd.DataFrame([summary])], ignore_index=True
    )

    csv_bytes = export_with_summary.to_csv(index=False).encode("utf-8")
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        export_with_summary.to_excel(writer, index=False)
    excel_buffer.seek(0)

    buttons = st.columns(2)
    with buttons[0]:
        st.download_button(
            "📤 Exportar CSV",
            data=csv_bytes,
            file_name="recomendaciones_inteligentes.csv",
            mime="text/csv",
        )
    with buttons[1]:
        st.download_button(
            "📥 Exportar XLSX",
            data=excel_buffer.getvalue(),
            file_name="recomendaciones_inteligentes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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
        st.plotly_chart(pie_fig, width="stretch", config={"displayModeBar": False})
    with charts[1]:
        st.plotly_chart(bar_fig, width="stretch", config={"displayModeBar": False})


def render_recommendations_tab() -> None:
    """Render the recommendations tab inside the main dashboard."""

    positions = _get_portfolio_positions()
    st.header("🔮 Recomendaciones personalizadas")
    st.caption(
        "Proyectá cómo complementar tu cartera con nuevas ideas balanceadas según tu perfil."
    )

    profile_service = ProfileService()
    active_profile = _render_profile_panel(profile_service)

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
    profile_from_state: dict[str, str] | None = None

    state_payload: dict[str, object] | None = None

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
        profile_candidate = stored_state.get("profile")
        if isinstance(profile_candidate, dict):
            profile_from_state = profile_candidate
    if profile_from_state:
        active_profile = profile_from_state

    symbols = [str(sym) for sym in positions.get("simbolo", []) if sym]

    with st.form(_FORM_KEY):
        amount_default = stored_amount if stored_amount and np.isfinite(stored_amount) else 100_000.0
        amount = st.number_input(
            "Monto disponible a invertir (ARS)",
            min_value=0.0,
            value=float(amount_default),
            step=10_000.0,
            format="%0.0f",
            key=f"{_FORM_KEY}_amount_input",
        )
        profile_default = (profile_from_state or active_profile).get("preferred_mode", "diversify")
        default_mode_key = stored_mode_key or profile_default or _MODE_OPTIONS[0][0]
        default_index = next(
            (idx for idx, option in enumerate(_MODE_OPTIONS) if option[0] == default_mode_key),
            0,
        )
        mode = st.selectbox(
            "Modo de recomendación",
            options=_MODE_OPTIONS,
            format_func=lambda item: item[1],
            index=default_index,
            key=f"{_FORM_KEY}_mode_select",
        )
        submitted = st.form_submit_button(
            "Calcular recomendaciones",
            key=f"{_FORM_KEY}_submit_button",
        )

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
            recommendations = svc.recommend(
                amount,
                mode=mode[0],
                profile=active_profile,
            )
            analysis_data = svc.analyze_portfolio()

            if recommendations.empty:
                st.session_state.pop(_SESSION_STATE_KEY, None)
            else:
                state_payload = {
                    "recommendations": recommendations.copy(),
                    "opportunities": opportunities.copy(),
                    "risk_metrics": risk_metrics.copy(),
                    "amount": amount,
                    "mode_label": mode[1],
                    "mode_key": mode[0],
                    "analysis": analysis_data,
                    "profile": active_profile.copy(),
                }
            stored_amount = amount
            stored_mode_label = mode[1]
            stored_mode_key = mode[0]

    if isinstance(analysis_data, dict):
        _render_analysis_summary(analysis_data)
    elif isinstance(stored_state.get("analysis"), dict):
        _render_analysis_summary(stored_state["analysis"])

    amount_to_display = amount
    if stored_amount is not None and np.isfinite(stored_amount):
        amount_to_display = stored_amount
    mode_key_to_display, derived_label = _resolve_mode(stored_mode_key or mode[0])
    mode_label_to_display = stored_mode_label or derived_label

    expected_map = _expected_return_map(opportunities)
    beta_lookup = _beta_lookup(risk_metrics, opportunities)

    recommendations = _enrich_recommendations(
        recommendations,
        expected_returns=expected_map,
        betas=beta_lookup,
    )

    expected_map = {
        **expected_map,
        **_build_numeric_lookup(recommendations, "expected_return"),
    }
    beta_lookup = {
        **beta_lookup,
        **_build_numeric_lookup(recommendations, "beta"),
    }

    if state_payload is not None:
        state_payload["recommendations"] = recommendations.copy()
        try:
            st.session_state[_SESSION_STATE_KEY] = state_payload
        except Exception:  # pragma: no cover - defensive safeguard
            LOGGER.debug("No se pudo guardar el estado de recomendaciones", exc_info=True)
    elif stored_state and isinstance(stored_state, dict):
        if profile_from_state != active_profile:
            updated_state = stored_state.copy()
            updated_state["profile"] = active_profile.copy()
            try:
                st.session_state[_SESSION_STATE_KEY] = updated_state
            except Exception:  # pragma: no cover - defensive safeguard
                LOGGER.debug(
                    "No se pudo actualizar el perfil en el estado de recomendaciones",
                    exc_info=True,
                )

    if not recommendations.empty:
        _render_recommendations_visuals(
            recommendations,
            mode_label=mode_label_to_display,
            amount=amount_to_display,
        )
        _render_benchmark_block(recommendations)
    show_predictions = st.toggle(
        "Incluir predicciones",
        value=True,
        key=f"{_FORM_KEY}_show_predictions_toggle",
    )
    stats = get_cache_stats()
    if stats.total:
        st.caption(
            "Cache de predicciones sectoriales: "
            f"{stats.hits}/{stats.total} hits ({stats.hit_ratio * 100:.1f}%)"
        )
    else:
        st.caption("Cache de predicciones sectoriales en calentamiento")
    _render_recommendations_table(
        recommendations,
        show_predictions=show_predictions,
    )
    _render_automatic_insight(
        recommendations,
        mode_key=mode_key_to_display,
        beta_lookup=beta_lookup,
    )

    simulate_key = f"simulate_button_{mode_key_to_display}" if mode_key_to_display else "simulate_button"
    simulate_clicked = st.button(
        "Simular impacto",
        disabled=recommendations.empty,
        key=simulate_key,
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


def _render_for_test(recommendations_df: pd.DataFrame, state: object) -> None:
    _suppress_streamlit_bare_mode_warnings()

    try:
        selected_mode = getattr(state, "selected_mode", "diversify")
    except Exception:
        selected_mode = "diversify"
    mode_key, mode_label = _resolve_mode(selected_mode)

    session = getattr(st, "session_state", {})
    if "portfolio_last_positions" not in session:
        st.session_state["portfolio_last_positions"] = pd.DataFrame(
            [{"simbolo": "TEST", "valor_actual": 100_000.0}]
        )
    if ProfileService.SESSION_KEY not in session:
        profile = DEFAULT_PROFILE.copy()
        fixture_path = Path("docs/fixtures/default/profile_default.json")
        if fixture_path.exists():
            try:
                raw_text = fixture_path.read_text(encoding="utf-8")
                payload = json.loads(raw_text) or {}
            except (OSError, json.JSONDecodeError):
                payload = {}
            if isinstance(payload, dict):
                overrides = {}
                for key in profile.keys():
                    value = payload.get(key)
                    if isinstance(value, str) and value:
                        overrides[key] = value
                profile.update(overrides)
                if "last_updated" in payload:
                    profile["last_updated"] = payload["last_updated"]
        st.session_state[ProfileService.SESSION_KEY] = profile

    warmup_frame = pd.DataFrame()
    if {"symbol", "sector"}.issubset(recommendations_df.columns):
        warmup_frame = recommendations_df[["symbol", "sector"]].dropna().copy()
    try:
        predict_sector_performance(warmup_frame)
        predict_sector_performance(warmup_frame)
    except Exception:  # pragma: no cover - defensive warmup
        LOGGER.debug("No se pudo precalentar predicciones sectoriales", exc_info=True)

    payload_df = recommendations_df.copy()
    if "predicted_return_pct" not in payload_df.columns:
        payload_df["predicted_return_pct"] = np.nan

    st.session_state[_SESSION_STATE_KEY] = {
        "recommendations": _enrich_recommendations(
            payload_df,
            expected_returns=_build_numeric_lookup(payload_df, "expected_return"),
            betas=_build_numeric_lookup(payload_df, "beta"),
        ),
        "opportunities": pd.DataFrame(),
        "risk_metrics": pd.DataFrame(),
        "amount": float(
            pd.to_numeric(payload_df.get("allocation_amount"), errors="coerce").sum()
        ),
        "mode_label": mode_label,
        "mode_key": mode_key,
        "analysis": {},
        "profile": DEFAULT_PROFILE.copy(),
    }

    from contextlib import ExitStack, redirect_stderr, redirect_stdout
    from io import StringIO

    with ExitStack() as stack:
        stack.enter_context(redirect_stdout(StringIO()))
        stack.enter_context(redirect_stderr(StringIO()))
        render_recommendations_tab()


__all__ = ["render_recommendations_tab", "_render_for_test"]
