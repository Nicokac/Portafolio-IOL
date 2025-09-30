from __future__ import annotations

"""Sidebar panel summarising recent data source health."""

from typing import Iterable, Optional

import streamlit as st

from services.health import get_health_metrics
from shared.time_provider import TimeProvider
from shared.ui import notes as shared_notes
from shared.version import __version__


def _format_timestamp(ts: Optional[float]) -> str:
    if not ts:
        return "Sin registro"
    snapshot = TimeProvider.from_timestamp(ts)
    if snapshot is None:
        return "Sin registro"
    return snapshot.text


def _format_iol_status(data: Optional[dict]) -> str:
    if not data:
        return "_Sin actividad registrada._"
    status = data.get("status")
    icon = "✅" if status == "success" else "⚠️"
    label = "Refresh correcto" if status == "success" else "Error al refrescar"
    ts = _format_timestamp(data.get("ts"))
    detail = data.get("detail")
    detail_txt = f" — {detail}" if detail else ""
    return f"{icon} {label} • {ts}{detail_txt}"


def _format_yfinance_status(data: Optional[dict]) -> str:
    if not data:
        return "_Sin consultas registradas._"
    source = data.get("source") or "desconocido"
    mapping = {
        "yfinance": ("✅", "Datos de Yahoo Finance"),
        "fallback": ("♻️", "Fallback local"),
        "error": ("⚠️", "Error o sin datos"),
    }
    icon, label = mapping.get(source, ("ℹ️", f"Fuente: {source}"))
    ts = _format_timestamp(data.get("ts"))
    detail = data.get("detail")
    detail_txt = f" — {detail}" if detail else ""
    return f"{icon} {label} • {ts}{detail_txt}"


def _format_fx_api_status(data: Optional[dict]) -> str:
    if not data:
        return "_Sin llamadas a la API FX._"
    status = data.get("status")
    icon = "✅" if status == "success" else "⚠️"
    label = "API FX OK" if status == "success" else "API FX con errores"
    ts = _format_timestamp(data.get("ts"))
    elapsed = data.get("elapsed_ms")
    elapsed_txt = f"{float(elapsed):.0f} ms" if isinstance(elapsed, (int, float)) else "s/d"
    detail = data.get("error")
    detail_txt = f" — {detail}" if detail else ""
    return f"{icon} {label} • {ts} ({elapsed_txt}){detail_txt}"


def _format_fx_cache_status(data: Optional[dict]) -> str:
    if not data:
        return "_Sin uso de caché registrado._"
    mode = data.get("mode")
    icon = "♻️" if mode == "hit" else "🔄"
    label = "Uso de caché" if mode == "hit" else "Actualización"
    ts = _format_timestamp(data.get("ts"))
    age = data.get("age")
    age_txt = f"{float(age):.0f}s" if isinstance(age, (int, float)) else "s/d"
    return f"{icon} {label} • {ts} (edad {age_txt})"


def _format_latency_line(label: str, data: Optional[dict]) -> str:
    if not data:
        return f"- {label}: sin registro"
    elapsed = data.get("elapsed_ms")
    elapsed_txt = f"{float(elapsed):.0f} ms" if isinstance(elapsed, (int, float)) else "s/d"
    parts = [f"- {label}: {elapsed_txt}"]
    source = data.get("source")
    if source:
        parts.append(f"fuente: {source}")
    count = data.get("count")
    if count is not None:
        parts.append(f"items: {count}")
    detail = data.get("detail")
    if detail:
        parts.append(detail)
    parts.append(_format_timestamp(data.get("ts")))
    return " • ".join(parts)


def _format_fx_section(api_data: Optional[dict], cache_data: Optional[dict]) -> Iterable[str]:
    lines = []
    lines.append(_format_fx_api_status(api_data))
    lines.append(_format_fx_cache_status(cache_data))
    return lines


def _format_latency_section(portfolio: Optional[dict], quotes: Optional[dict]) -> Iterable[str]:
    return [
        _format_latency_line("Portafolio", portfolio),
        _format_latency_line("Cotizaciones", quotes),
    ]


def render_health_sidebar() -> None:
    """Render the health summary panel inside the sidebar."""
    metrics = get_health_metrics()
    sidebar = st.sidebar
    sidebar.header(f"🩺 Healthcheck (versión {__version__})")
    sidebar.caption("Monitorea la procedencia y el rendimiento de los datos cargados.")

    sidebar.markdown("#### 🔐 Conexión IOL")
    sidebar.markdown(
        shared_notes.format_note(_format_iol_status(metrics.get("iol_refresh")))
    )

    sidebar.markdown("#### 📈 Yahoo Finance")
    sidebar.markdown(
        shared_notes.format_note(_format_yfinance_status(metrics.get("yfinance")))
    )

    sidebar.markdown("#### 💱 FX")
    for line in _format_fx_section(metrics.get("fx_api"), metrics.get("fx_cache")):
        sidebar.markdown(shared_notes.format_note(line))

    sidebar.markdown("#### ⏱️ Latencias")
    for line in _format_latency_section(metrics.get("portfolio"), metrics.get("quotes")):
        sidebar.markdown(shared_notes.format_note(line))


__all__ = ["render_health_sidebar"]
