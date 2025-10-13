from __future__ import annotations

"""Streamlit-facing controller helpers for the portfolio dashboard."""

from typing import Any, Callable

import streamlit as st

from controllers.portfolio.portfolio import render_portfolio_section
from services.performance_timer import performance_timer
from services.performance_metrics import measure_execution


def render_portfolio_ui(
    container: Any,
    cli: Any,
    fx_rates: Any,
    *,
    view_model_service_factory: Callable[[], Any] | None = None,
    notifications_service_factory: Callable[[], Any] | None = None,
) -> Any:
    """Render the portfolio tab while measuring UI latency."""

    telemetry: dict[str, object] = {
        "status": "success",
        "has_cli": cli is not None,
    }
    active_tab = st.session_state.get("active_tab")
    if isinstance(active_tab, str):
        telemetry["active_tab"] = active_tab
    stage_timings: dict[str, float] = {}
    with performance_timer("render_portfolio_ui", extra=telemetry):
        try:
            with measure_execution("portfolio_ui.total"):
                refresh_secs = render_portfolio_section(
                    container,
                    cli,
                    fx_rates,
                    view_model_service_factory=view_model_service_factory,
                    notifications_service_factory=notifications_service_factory,
                    timings=stage_timings,
                )
        except Exception:
            telemetry["status"] = "error"
            raise
        if stage_timings:
            telemetry["timings_ms"] = stage_timings
        if isinstance(refresh_secs, (int, float)):
            telemetry["refresh_secs"] = float(refresh_secs)
        render_cache = st.session_state.get("render_cache")
        if isinstance(render_cache, dict):
            telemetry["cached_tabs"] = len(render_cache)
        return refresh_secs


__all__ = ["render_portfolio_ui"]
