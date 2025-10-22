from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd
import streamlit as st

from application.portfolio_service import PortfolioService

from .formatting import (
    _format_currency,
    _format_currency_delta,
    _format_float,
    _format_float_delta,
    _format_percent,
    _format_percent_delta,
)

LOGGER = logging.getLogger(__name__)

__all__ = ["render_simulation_panel", "_render_simulation_results"]


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
        st.caption(f"Se asignan {_format_currency(additional)} adicionales siguiendo la distribución sugerida.")


def render_simulation_panel(
    recommendations: pd.DataFrame,
    positions: pd.DataFrame,
    expected_map: Mapping[str, float],
    beta_lookup: Mapping[str, float],
    *,
    mode_key: str,
) -> None:
    simulate_key = f"simulate_button_{mode_key}" if mode_key else "simulate_button"
    simulate_clicked = st.button(
        "Simular impacto",
        disabled=recommendations.empty,
        key=simulate_key,
    )

    if not simulate_clicked:
        return

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
        return

    _render_simulation_results(result)
