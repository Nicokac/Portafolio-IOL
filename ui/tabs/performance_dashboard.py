from __future__ import annotations

"""Streamlit tab that visualises recent performance telemetry."""

from collections import defaultdict

import pandas as pd
import streamlit as st

from services.performance_timer import LOG_PATH, read_recent_entries


def _entries_to_dataframe(entries):
    rows: list[dict[str, object]] = []
    for entry in entries:
        row: dict[str, object] = {
            "Timestamp": entry.timestamp,
            "Bloque": entry.label,
            "Duración (s)": round(entry.duration_s, 4),
            "CPU (%)": None if entry.cpu_percent is None else round(entry.cpu_percent, 2),
            "RAM (%)": None if entry.ram_percent is None else round(entry.ram_percent, 2),
        }
        for key, value in entry.extras.items():
            row[f"extra:{key}"] = value
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["Timestamp", "Bloque", "Duración (s)"])
    return pd.DataFrame(rows)


def _build_average_summary(entries):
    aggregates: dict[str, list[float]] = defaultdict(list)
    for entry in entries:
        aggregates[entry.label].append(entry.duration_s)
    summary_rows = [
        {"Bloque": label, "Duración promedio (s)": sum(values) / len(values)}
        for label, values in aggregates.items()
        if values
    ]
    if not summary_rows:
        return pd.DataFrame(columns=["Bloque", "Duración promedio (s)"])
    frame = pd.DataFrame(summary_rows)
    return frame.sort_values(by="Duración promedio (s)", ascending=False)


def render_performance_dashboard_tab(limit: int = 200) -> None:
    """Render the performance telemetry dashboard for QA and diagnostics."""

    st.header("⏱️ Diagnóstico de performance")
    st.caption(
        "Explorá la telemetría generada por `services.performance_timer` para detectar cuellos de botella."
    )

    if st.button("🔄 Actualizar métricas", key="refresh_performance_dashboard"):
        st.experimental_rerun()

    entries = read_recent_entries(limit=limit)
    if not entries:
        st.info(
            "Todavía no se registraron mediciones. Ejecutá un flujo de portafolio o predicciones para generar datos."
        )
        st.caption(f"Archivo de log: {LOG_PATH}")
        return

    df = _entries_to_dataframe(entries)
    st.subheader("Registros recientes")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Promedio por bloque")
    summary = _build_average_summary(entries)
    if not summary.empty:
        summary_display = summary.set_index("Bloque")
        st.bar_chart(summary_display)
    else:
        st.caption("No hay suficientes datos para calcular promedios.")

    st.caption(f"Archivo de log: {LOG_PATH}")


__all__ = ["render_performance_dashboard_tab"]
