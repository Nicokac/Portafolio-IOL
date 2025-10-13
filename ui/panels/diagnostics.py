from __future__ import annotations

import pandas as pd
import streamlit as st
from typing import Callable

from application.predictive_service import get_cache_stats
from services.performance_metrics import export_metrics_csv, get_recent_metrics
from ui.helpers.navigation import safe_page_link
from ui.tabs.performance_dashboard import render_performance_dashboard_tab


def _format_memory(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _format_stage_label(name: str) -> str:
    """Return a human-friendly label for portfolio UI metric names."""

    stage = name.split("portfolio_ui.", 1)[-1]
    stage = stage.replace("render_tab.", "render_tab · ")
    stage = stage.replace(".", " · ")
    stage = stage.replace("_", " ")
    return stage.strip().title()


def _metrics_to_dataframe(
    summaries: list["MetricSummary"],
    *,
    name_label: str = "Función",
    name_transform: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    rows = [
        {
            name_label: name_transform(summary.name)
            if name_transform
            else summary.name,
            "Muestras": summary.samples,
            "Promedio (ms)": round(summary.average_ms, 2),
            "Último (ms)": round(summary.last_ms, 2),
            "Memoria prom. (KB)": _format_memory(summary.average_memory_kb),
            "Memoria última (KB)": _format_memory(summary.last_memory_kb),
            "Última ejecución": summary.last_run_iso,
        }
        for summary in summaries
    ]
    return pd.DataFrame(rows)


def render_diagnostics_panel() -> None:
    """Render diagnostic metrics for QA and support teams."""

    st.header("🩺 Diagnóstico de rendimiento")

    metrics = get_recent_metrics()
    portfolio_metrics = [m for m in metrics if m.name.startswith("portfolio_ui.")]
    remaining_metrics = [m for m in metrics if m not in portfolio_metrics]

    if portfolio_metrics:
        st.subheader("⏱️ Portfolio UI (subcomponentes)")
        portfolio_frame = _metrics_to_dataframe(
            portfolio_metrics,
            name_label="Subcomponente",
            name_transform=_format_stage_label,
        )
        st.dataframe(portfolio_frame, use_container_width=True, hide_index=True)

    if remaining_metrics:
        st.subheader("📈 Métricas instrumentadas")
        frame = _metrics_to_dataframe(remaining_metrics)
        st.dataframe(frame, use_container_width=True, hide_index=True)
    if not metrics:
        st.caption(
            "Aún no hay mediciones registradas. Ejecutá una simulación o predicción para generar datos."
        )

    snapshot = get_cache_stats()
    st.subheader("📦 Estado del caché predictivo")
    cols = st.columns(3)
    cols[0].metric("Hits", snapshot.hits)
    cols[1].metric("Misses", snapshot.misses)
    cols[2].metric("Hit ratio", f"{snapshot.hit_ratio * 100:.1f}%")
    ttl_label = f"TTL efectivo: {snapshot.ttl_hours:.1f} h" if snapshot.ttl_hours is not None else "TTL efectivo: s/d"
    last_label = snapshot.last_updated or "-"
    st.caption(f"Última actualización: {last_label} · {ttl_label}")

    csv_payload = export_metrics_csv()
    st.download_button(
        "⬇️ Exportar métricas (CSV)",
        data=csv_payload,
        file_name="performance_metrics.csv",
        mime="text/csv",
        disabled=not metrics,
    )

    st.divider()
    st.subheader("⏱️ Telemetría detallada")
    safe_page_link(
        "ui.tabs.performance_dashboard",
        label="Abrir dashboard de performance",
        render_fallback=render_performance_dashboard_tab,
    )


__all__ = ["render_diagnostics_panel"]

