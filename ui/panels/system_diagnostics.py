from __future__ import annotations

import pandas as pd
import streamlit as st

from services.system_diagnostics import get_system_diagnostics_snapshot


def _format_ms(value: float | None) -> str:
    if value is None:
        return "s/d"
    return f"{value:.2f}"


def _format_status(valid: bool, is_weak: bool | None) -> str:
    if not valid:
        return "⚠️ Revisar"
    if is_weak:
        return "⚠️ Débil"
    return "✅ Correcta"


def render_system_diagnostics_panel() -> None:
    """Render aggregated system diagnostics within the UI."""

    snapshot = get_system_diagnostics_snapshot()

    st.header("🔎 Diagnóstico del sistema")
    st.caption("Benchmarks periódicos sobre endpoints críticos y salud operativa.")

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
        st.dataframe(frame, use_container_width=True, hide_index=True)
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
    st.dataframe(pd.DataFrame(key_rows), use_container_width=True, hide_index=True)

    env = snapshot.environment
    st.markdown(
        f"""
        <div style="padding:0.8rem 1rem; border-radius:0.75rem; background:rgba(15,23,42,0.05);">
            <div style="display:flex; flex-wrap:wrap; gap:1.2rem; align-items:center;">
                <div><strong>APP_ENV</strong><br>{env.app_env or 's/d'}</div>
                <div><strong>Zona horaria</strong><br>{env.timezone or 's/d'}</div>
                <div><strong>Python</strong><br>{env.python_version}</div>
                <div><strong>Plataforma</strong><br>{env.platform}</div>
            </div>
            <p style="margin:0.6rem 0 0; font-size:0.85rem; color:rgba(15,23,42,0.75);">
                Los fingerprints muestran los primeros 10 caracteres del hash SHA-1 de cada clave para identificar cambios sin exponer su valor.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


__all__ = ["render_system_diagnostics_panel"]
