from __future__ import annotations

import json
import logging
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from application.adaptive_predictive_service import (
    export_adaptive_report,
    generate_synthetic_history,
    prepare_adaptive_history,
    simulate_adaptive_forecast,
)
from application.predictive_service import get_cache_stats, predict_sector_performance
from application.profile_service import DEFAULT_PROFILE, ProfileService
from shared.logging_utils import silence_streamlit_warnings
from ui.charts.correlation_matrix import build_correlation_figure

from .recommendations import (
    _SESSION_STATE_KEY,
    _build_numeric_lookup,
    _enrich_recommendations,
    _resolve_mode,
    render_recommendations_tab,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "render_recommendations_tab",
    "_render_for_test",
    "prepare_adaptive_history",
    "generate_synthetic_history",
    "simulate_adaptive_forecast",
    "export_adaptive_report",
    "build_correlation_figure",
    "get_cache_stats",
    "px",
]


def _render_for_test(recommendations_df: pd.DataFrame, state: object) -> None:
    silence_streamlit_warnings()

    try:
        selected_mode = getattr(state, "selected_mode", "diversify")
    except Exception:
        selected_mode = "diversify"
    mode_key, mode_label = _resolve_mode(selected_mode)

    if not isinstance(recommendations_df, pd.DataFrame) or recommendations_df.empty:
        recommendations_df = pd.DataFrame(
            [
                {
                    "symbol": "TEST",
                    "sector": "Tecnología",
                    "predicted_return_pct": 4.2,
                    "expected_return": 3.8,
                },
                {
                    "symbol": "ALT",
                    "sector": "Finanzas",
                    "predicted_return_pct": 3.4,
                    "expected_return": 2.9,
                },
            ]
        )

    if "sector" not in recommendations_df.columns:
        recommendations_df["sector"] = ["Tecnología", "Finanzas", "Energía"][: len(recommendations_df)]

    with ExitStack() as stack:
        stack.enter_context(redirect_stdout(StringIO()))
        stack.enter_context(redirect_stderr(StringIO()))

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
        if pd.to_numeric(payload_df.get("predicted_return_pct"), errors="coerce").isna().all():
            payload_df["predicted_return_pct"] = np.linspace(3.0, 4.5, len(payload_df))

        st.session_state[_SESSION_STATE_KEY] = {
            "recommendations": _enrich_recommendations(
                payload_df,
                expected_returns=_build_numeric_lookup(payload_df, "expected_return"),
                betas=_build_numeric_lookup(payload_df, "beta"),
            ),
            "opportunities": pd.DataFrame(),
            "risk_metrics": pd.DataFrame(),
            "amount": float(pd.to_numeric(payload_df.get("allocation_amount"), errors="coerce").sum()),
            "mode_label": mode_label,
            "mode_key": mode_key,
            "analysis": {},
            "profile": DEFAULT_PROFILE.copy(),
        }

        render_recommendations_tab()
