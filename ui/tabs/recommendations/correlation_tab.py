from __future__ import annotations

import logging
from typing import Mapping

import pandas as pd
import streamlit as st

from application.adaptive_predictive_service import export_adaptive_report
from controllers import recommendations_controller
from ui.charts.correlation_matrix import build_correlation_figure

from .formatting import (
    _format_float,
    _format_percent,
    _format_percent_delta,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "_compute_adaptive_payload",
    "_render_correlation_tab",
]


def _merge_symbol_sector_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for frame in frames:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        df = frame.copy()
        if "symbol" not in df.columns and "ticker" in df.columns:
            df = df.rename(columns={"ticker": "symbol"})
        if {"symbol", "sector"}.issubset(df.columns):
            df["symbol"] = df.get("symbol", pd.Series(dtype=str)).astype("string").str.upper().str.strip()
            df["sector"] = (
                df.get("sector", pd.Series(dtype=str)).astype("string").str.strip().replace({"": "Sin sector"})
            )
            rows.append(df[["symbol", "sector"]])
    if not rows:
        return pd.DataFrame(columns=["symbol", "sector"])
    merged = pd.concat(rows, ignore_index=True)
    merged = merged.dropna(subset=["symbol", "sector"])
    merged = merged.drop_duplicates(subset=["symbol"])
    return merged


def _compute_adaptive_payload(
    recommendations: pd.DataFrame,
    opportunities: pd.DataFrame,
    *,
    profile: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    if not isinstance(recommendations, pd.DataFrame) or recommendations.empty:
        return None

    opportunity_frame = _merge_symbol_sector_frames(recommendations, opportunities)
    history, synthetic = recommendations_controller.build_adaptive_history_view(
        opportunity_frame,
        recommendations,
        profile=profile,
    )
    if history.empty:
        return None

    try:
        forecast_view = recommendations_controller.run_adaptive_forecast_view(
            history,
            ema_span=4,
            persist=not synthetic,
            context={
                "profile": profile,
                "symbols": opportunity_frame.get("symbol").tolist()
                if isinstance(opportunity_frame, pd.DataFrame)
                else [],
                "sectors": opportunity_frame.get("sector").tolist()
                if isinstance(opportunity_frame, pd.DataFrame)
                else [],
            },
        )
    except Exception:  # pragma: no cover - defensive logging
        LOGGER.exception("No se pudo calcular el modelo adaptativo", exc_info=True)
        return None

    payload = forecast_view.payload
    payload["history_frame"] = history
    payload["synthetic"] = synthetic
    return payload


def _render_correlation_tab(payload: Mapping[str, object] | None) -> None:
    if not isinstance(payload, Mapping) or not payload:
        st.info("No hay correlaciones adaptativas disponibles todavía.")
        return

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    beta_shift = payload.get("beta_shift")
    historical = payload.get("historical_correlation")
    rolling = payload.get("rolling_correlation")
    adaptive = payload.get("correlation_matrix")
    metadata = payload.get("cache_metadata") if isinstance(payload.get("cache_metadata"), dict) else {}

    hit_ratio = float(metadata.get("hit_ratio", 0.0)) if metadata else 0.0
    last_updated_label = str(metadata.get("last_updated", "-")) if metadata else "-"
    st.caption(
        f"Última actualización del estado adaptativo: {last_updated_label} | Ratio de aciertos: {hit_ratio:.0f} %"
    )

    cols = st.columns(4)
    beta_value = float(summary.get("beta_mean", float("nan")))
    corr_value = float(summary.get("correlation_mean", float("nan")))
    sigma_value = float(summary.get("sector_dispersion", float("nan")))
    beta_shift_avg = float(summary.get("beta_shift_avg", float("nan")))

    cols[0].metric("β promedio", _format_float(beta_value))
    cols[1].metric("Correlación media", _format_float(corr_value))
    cols[2].metric(
        "Dispersión sectorial σ",
        _format_float(sigma_value),
        help="Dispersión sectorial (volatilidad entre sectores pronosticados)",
    )
    cols[3].metric(
        "β-shift promedio",
        _format_float(beta_shift_avg),
        help="Cambio promedio en sensibilidad del portafolio respecto al benchmark",
    )

    if payload.get("synthetic"):
        st.caption("Usando histórico sintético hasta contar con mediciones reales.")

    if st.button("Exportar reporte adaptativo", key="export_adaptive_report"):
        st.toast("Generando reporte adaptativo...")
        try:
            report_path = export_adaptive_report(payload)
        except Exception:  # pragma: no cover - UX feedback
            LOGGER.exception("No se pudo exportar el reporte adaptativo")
            st.toast("❌ Error al exportar reporte")
            st.error("No se pudo exportar el reporte adaptativo. Intentá nuevamente.")
        else:
            st.toast("✅ Reporte generado")
            st.success(f"Reporte generado en {report_path}")

    figure = build_correlation_figure(
        historical,
        rolling,
        adaptive,
        beta_shift=beta_shift if isinstance(beta_shift, pd.Series) else None,
        title="Correlaciones sectoriales β-shift",
    )
    st.plotly_chart(figure, width="stretch", config={"responsive": True})

    mae = float(summary.get("mae", 0.0))
    rmse = float(summary.get("rmse", 0.0))
    bias = float(summary.get("bias", 0.0))
    raw_mae = float(summary.get("raw_mae", 0.0))
    raw_rmse = float(summary.get("raw_rmse", 0.0))
    raw_bias = float(summary.get("raw_bias", 0.0))

    metrics_cols = st.columns(3)
    metrics_cols[0].metric(
        "MAE adaptativo",
        _format_percent(mae),
        _format_percent_delta(raw_mae - mae),
    )
    metrics_cols[1].metric(
        "RMSE adaptativo",
        _format_percent(rmse),
        _format_percent_delta(raw_rmse - rmse),
    )
    metrics_cols[2].metric(
        "Bias",
        _format_percent(bias),
        _format_percent_delta(raw_bias - bias),
    )

    steps = payload.get("steps")
    if isinstance(steps, pd.DataFrame) and not steps.empty:
        preview = steps.copy().sort_values("timestamp", ascending=False).head(6)
        preview["timestamp"] = pd.to_datetime(preview["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d")
        st.dataframe(preview, width="stretch", hide_index=True)
