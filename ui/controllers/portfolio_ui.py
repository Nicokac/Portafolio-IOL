"""Streamlit-facing controller helpers for the portfolio dashboard."""

import importlib
from functools import lru_cache
from typing import Any, Callable

import streamlit as st

from services import snapshot_defer
from services.environment import mark_portfolio_ui_render_complete
from services.performance_metrics import measure_execution


@lru_cache(maxsize=1)
def _get_portfolio_section():
    module = importlib.import_module("controllers.portfolio.portfolio")
    return getattr(module, "render_portfolio_section")


@lru_cache(maxsize=1)
def _get_performance_timer():
    module = importlib.import_module("services.performance_timer")
    return getattr(module, "performance_timer")


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
    st.session_state["ui_idle"] = False
    snapshot_defer.mark_ui_busy()
    active_tab = st.session_state.get("active_tab")
    if isinstance(active_tab, str):
        telemetry["active_tab"] = active_tab
    tab_loaded = st.session_state.get("tab_loaded")
    if not isinstance(tab_loaded, dict):
        tab_loaded = {}
        st.session_state["tab_loaded"] = tab_loaded
    stage_timings: dict[str, float] = {}
    portfolio_section = _get_portfolio_section()
    performance_timer = _get_performance_timer()

    with performance_timer("render_portfolio_ui", extra=telemetry):
        try:
            with measure_execution("portfolio_ui.total"):
                refresh_secs = portfolio_section(
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
            try:
                st.session_state["portfolio_stage_timings"] = dict(stage_timings)
            except Exception:  # pragma: no cover - defensive safeguard
                pass
        if isinstance(refresh_secs, (int, float)):
            telemetry["refresh_secs"] = float(refresh_secs)
        render_cache = st.session_state.get("render_cache")
        if isinstance(render_cache, dict):
            telemetry["cached_tabs"] = len(render_cache)
        try:
            dataset_hash = st.session_state.get("dataset_hash")
        except Exception:  # pragma: no cover - defensive safeguard
            dataset_hash = None
        if isinstance(dataset_hash, str) and dataset_hash:
            telemetry["dataset_hash"] = dataset_hash
        reused_visual_cache = bool(
            st.session_state.get("__portfolio_visual_cache_reused__")
        )
        telemetry["reused_visual_cache"] = reused_visual_cache
        visual_cache = st.session_state.get("cached_render")
        if isinstance(visual_cache, dict):
            telemetry["visual_cache_datasets"] = len(visual_cache)
        mark_portfolio_ui_render_complete()
        st.session_state["ui_idle"] = True
        snapshot_defer.mark_ui_idle()
        return refresh_secs


__all__ = ["render_portfolio_ui"]
