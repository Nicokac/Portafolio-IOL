from __future__ import annotations

import pandas as pd
import streamlit as st
from typing import Callable

from application.predictive_service import get_cache_stats
from services.performance_metrics import export_metrics_csv, get_recent_metrics
from ui.helpers.navigation import safe_page_link
from ui.tabs.performance_dashboard import render_performance_dashboard_tab
from shared.time_provider import TimeProvider


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

    stage_timings = st.session_state.get("portfolio_stage_timings")
    fingerprint_stats = st.session_state.get("portfolio_fingerprint_cache_stats")
    total_load_ms = st.session_state.get("total_load_ms")
    timing_rows: list[dict[str, object]] = []
    if isinstance(stage_timings, dict) and stage_timings:
        timing_rows.extend(
            {
                "Subetapa": _format_stage_label(f"portfolio_ui.{name}"),
                "Duración (ms)": round(float(value), 2),
            }
            for name, value in sorted(stage_timings.items())
        )
    include_total_load_row = isinstance(total_load_ms, (int, float)) and (
        not isinstance(stage_timings, dict) or "total_ms" not in stage_timings
    )
    if include_total_load_row:
        timing_rows.insert(
            0,
            {
                "Subetapa": _format_stage_label("portfolio_ui.total_ms"),
                "Duración (ms)": round(float(total_load_ms), 2),
            },
        )
    if timing_rows:
        st.subheader("🧭 Última renderización del portafolio")
        timings_frame = pd.DataFrame(timing_rows)
        st.dataframe(timings_frame, use_container_width=True, hide_index=True)

    if isinstance(fingerprint_stats, dict) and fingerprint_stats:
        hits = int(fingerprint_stats.get("hits", 0) or 0)
        misses = int(fingerprint_stats.get("misses", 0) or 0)
        ratio = float(fingerprint_stats.get("hit_ratio", 0.0) or 0.0) * 100.0
        last_status = str(fingerprint_stats.get("last_status") or "").strip().title()
        last_latency = float(fingerprint_stats.get("last_latency_ms", 0.0) or 0.0)
        last_key = str(fingerprint_stats.get("last_key") or "").strip()
        st.subheader("🔐 Caché de fingerprint de portafolio")
        cols = st.columns(3)
        cols[0].metric("Hits", hits)
        cols[1].metric("Misses", misses)
        cols[2].metric("Hit ratio", f"{ratio:.1f}%")
        caption_bits = []
        if last_status:
            caption_bits.append(f"Último acceso: {last_status}")
        if last_latency:
            caption_bits.append(f"Latencia: {last_latency:.2f} ms")
        if last_key:
            caption_bits.append(f"Clave: {last_key}")
        if caption_bits:
            st.caption(" · ".join(caption_bits))
    elif fingerprint_stats is not None:
        st.subheader("🔐 Caché de fingerprint de portafolio")
        st.caption("Aún no hay métricas registradas para el caché de fingerprints.")

    metrics = get_recent_metrics()
    portfolio_metrics = [m for m in metrics if m.name.startswith("portfolio_ui.")]
    remaining_metrics = [m for m in metrics if m not in portfolio_metrics]

    portfolio_frame = None
    if portfolio_metrics:
        portfolio_frame = _metrics_to_dataframe(
            portfolio_metrics,
            name_label="Subcomponente",
            name_transform=_format_stage_label,
        )

    if isinstance(total_load_ms, (int, float)):
        load_row = pd.DataFrame(
            [
                {
                    "Subcomponente": _format_stage_label("portfolio_ui.total_ms"),
                    "Muestras": 1,
                    "Promedio (ms)": round(float(total_load_ms), 2),
                    "Último (ms)": round(float(total_load_ms), 2),
                    "Memoria prom. (KB)": "-",
                    "Memoria última (KB)": "-",
                    "Última ejecución": TimeProvider.now(),
                }
            ]
        )
        if portfolio_frame is None:
            portfolio_frame = load_row
        else:
            portfolio_frame = pd.concat([load_row, portfolio_frame], ignore_index=True)

    if portfolio_frame is not None and not portfolio_frame.empty:
        st.subheader("⏱️ Portfolio UI (subcomponentes)")
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

