from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from services.system_diagnostics import get_system_diagnostics_snapshot
from shared import telemetry


def _format_ms(value: float | None) -> str:
    if value is None:
        return "s/d"
    return f"{value:.2f}"


def _format_metric_value(value: float | int | None) -> str:
    if value is None:
        return "s/d"
    try:
        numeric = float(value)
    except Exception:
        return "s/d"
    return f"{numeric:.0f} ms"


def _format_status(valid: bool, is_weak: bool | None) -> str:
    if not valid:
        return "⚠️ Revisar"
    if is_weak:
        return "⚠️ Débil"
    return "✅ Correcta"


def render_system_diagnostics_panel() -> None:
    """Render aggregated system diagnostics within the UI."""

    snapshot = get_system_diagnostics_snapshot()
    qa_metrics = _load_qa_metrics()

    st.header("🔎 Diagnóstico del sistema")
    st.caption("Benchmarks periódicos sobre endpoints críticos y salud operativa.")

    version = snapshot.version
    st.caption(
        f"Versión v{version.version} · Build {version.build_signature} · Release {version.release_date or 's/d'}"
    )

    try:
        startup_ms = st.session_state.get("ui_startup_load_ms")
        total_ms = st.session_state.get("total_load_ms")
    except Exception:
        startup_ms = None
        total_ms = None

    st.subheader("🕒 Tiempos de arranque")
    cols = st.columns(2)
    cols[0].metric("ui_startup_load_ms", _format_metric_value(startup_ms))
    cols[1].metric("ui_total_load_ms", _format_metric_value(total_ms))
    st.caption(
        "`ui_startup_load_ms` refleja el tiempo hasta que el login queda interactivo;"
        " `ui_total_load_ms` cubre el render completo tras autenticar."
    )

    st.subheader("📈 Memoria y carga QA")
    _render_memory_diagnostics(qa_metrics)

    st.subheader("📊 Métricas QA promedio")
    _render_qa_metrics_table(qa_metrics)

    st.subheader("⏱️ Latencias promedio")
    if snapshot.endpoints:
        rows = []
        for entry in snapshot.endpoints:
            rows.append(
                {
                    "Endpoint": entry.name,
                    "Promedio (ms)": _format_ms(entry.average_ms),
                    "Base histórica (ms)": _format_ms(entry.baseline_ms),
                    "Último (ms)": _format_ms(entry.last_ms),
                    "Muestras": entry.samples,
                    "Estado": "⚠️ Degradado" if entry.degraded else "✅ Estable",
                }
            )
        frame = pd.DataFrame(rows)
        st.dataframe(frame, width="stretch", hide_index=True)
    else:
        st.caption("No hay métricas registradas todavía. Ejecutá flujos para generar datos.")

    st.subheader("📦 Salud del caché predictivo")
    if snapshot.cache is not None:
        cols = st.columns(3)
        cols[0].metric("Hits", snapshot.cache.hits)
        cols[1].metric("Misses", snapshot.cache.misses)
        cols[2].metric("Hit ratio", f"{snapshot.cache.hit_ratio * 100:.1f}%")
        parts = [f"Última actualización: {snapshot.cache.last_updated}"]
        if snapshot.cache.ttl_hours is not None:
            parts.append(f"TTL efectivo: {snapshot.cache.ttl_hours:.1f} h")
        if snapshot.cache.remaining_ttl is not None:
            parts.append(f"TTL restante: {snapshot.cache.remaining_ttl / 3600:.1f} h")
        st.caption(" · ".join(parts))
    else:
        st.caption("El caché no reporta métricas en este momento.")

    st.subheader("🔐 Claves Fernet y entorno")
    key_rows = []
    for key in snapshot.keys:
        key_rows.append(
            {
                "Variable": key.name,
                "Estado": _format_status(key.valid, key.is_weak),
                "Fingerprint": key.fingerprint or "—",
                "Detalle": key.detail or ("Clave con baja entropía" if key.is_weak else "—"),
            }
        )
    st.dataframe(pd.DataFrame(key_rows), width="stretch", hide_index=True)

    env = snapshot.environment
    st.markdown(
        f"""
        <div style="padding:0.8rem 1rem; border-radius:0.75rem; background:rgba(15,23,42,0.05);">
            <div style="display:flex; flex-wrap:wrap; gap:1.2rem; align-items:center;">
                <div><strong>APP_ENV</strong><br>{env.app_env or "s/d"}</div>
                <div><strong>Zona horaria</strong><br>{env.timezone or "s/d"}</div>
                <div><strong>Python</strong><br>{env.python_version}</div>
                <div><strong>Plataforma</strong><br>{env.platform}</div>
            </div>
            <p style="margin:0.6rem 0 0; font-size:0.85rem; color:rgba(15,23,42,0.75);">
                Los fingerprints muestran los primeros 10 caracteres del hash SHA-1
                de cada clave para identificar cambios sin exponer su valor.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


__all__ = ["render_system_diagnostics_panel"]


def _load_qa_metrics() -> pd.DataFrame:
    path = telemetry._QA_METRICS_FILE
    if not isinstance(path, Path):
        path = Path(path)
    try:
        if not path.exists():
            return pd.DataFrame(columns=telemetry._QA_METRIC_COLUMNS)
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=telemetry._QA_METRIC_COLUMNS)
    return frame


def _render_memory_diagnostics(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.caption("No hay métricas QA registradas todavía.")
        return

    column = "peak_ram_mb"
    if column not in frame.columns:
        st.caption("El archivo de métricas QA no contiene mediciones de memoria.")
        return

    timestamp_series = pd.to_datetime(frame.get("timestamp"), errors="coerce")
    memory_series = pd.to_numeric(frame[column], errors="coerce")
    chart_data = pd.DataFrame(
        {
            "timestamp": timestamp_series,
            "peak_ram_mb": memory_series,
        }
    ).dropna()

    if chart_data.empty:
        st.caption("Aún no hay registros de memoria para graficar.")
        return

    fig = px.bar(
        chart_data,
        x="timestamp",
        y="peak_ram_mb",
        labels={"timestamp": "Evento", "peak_ram_mb": "Memoria máxima (MB)"},
        title="Uso de memoria por evento QA",
    )
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _render_qa_metrics_table(frame: pd.DataFrame) -> None:
    metrics: list[dict[str, object]] = []
    for field in telemetry._QA_METRIC_COLUMNS[2:]:
        series = pd.to_numeric(frame.get(field), errors="coerce") if field in frame else None
        if series is None or series.dropna().empty:
            display_value: object = "s/d"
        else:
            average = float(series.dropna().mean())
            display_value = f"{average:.3f}"
        metrics.append({"Métrica": field, "Promedio": display_value})

    if not metrics:
        st.caption("No hay métricas QA disponibles para resumir.")
        return

    st.dataframe(pd.DataFrame(metrics), width="stretch", hide_index=True)
