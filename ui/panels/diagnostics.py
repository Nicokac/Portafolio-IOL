from __future__ import annotations

import pandas as pd
import streamlit as st

from application.predictive_service import get_cache_stats
from services.performance_metrics import export_metrics_csv, get_recent_metrics


def _format_memory(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def render_diagnostics_panel() -> None:
    """Render diagnostic metrics for QA and support teams."""

    st.header("🩺 Diagnóstico de rendimiento")

    metrics = get_recent_metrics()
    if metrics:
        rows = [
            {
                "Función": summary.name,
                "Muestras": summary.samples,
                "Promedio (ms)": round(summary.average_ms, 2),
                "Último (ms)": round(summary.last_ms, 2),
                "Memoria prom. (KB)": _format_memory(summary.average_memory_kb),
                "Memoria última (KB)": _format_memory(summary.last_memory_kb),
                "Última ejecución": summary.last_run_iso,
            }
            for summary in metrics
        ]
        frame = pd.DataFrame(rows)
        st.dataframe(frame, use_container_width=True, hide_index=True)
    else:
        st.caption("Aún no hay mediciones registradas. Ejecutá una simulación o predicción para generar datos.")

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


__all__ = ["render_diagnostics_panel"]

