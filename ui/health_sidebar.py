from __future__ import annotations

"""Sidebar panel summarising recent data source health."""

from typing import Any, Iterable, Mapping, Optional, Sequence

import streamlit as st

from services.health import get_health_metrics
from shared.time_provider import TimeProvider
from shared.ui import notes as shared_notes
from shared.version import __version__

format_note = shared_notes.format_note


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
    label = "Refresh correcto" if status == "success" else "Error al refrescar"
    ts = _format_timestamp(data.get("ts"))
    detail = data.get("detail")
    detail_txt = f" — {detail}" if detail else ""
    prefix = "✅" if status == "success" else "⚠️"
    return format_note(f"{prefix} {label} • {ts}{detail_txt}")


def _format_yfinance_status(data: Optional[dict]) -> str:
    if not data:
        return "_Sin consultas registradas._"
    source = data.get("source") or "desconocido"
    mapping = {
        "yfinance": ("✅", "Datos de Yahoo Finance"),
        "fallback": ("ℹ️", "Fallback local"),
        "error": ("⚠️", "Error o sin datos"),
    }
    icon, label = mapping.get(source, ("ℹ️", f"Fuente: {source}"))
    ts = _format_timestamp(data.get("ts"))
    detail = data.get("detail")
    detail_txt = f" — {detail}" if detail else ""
    return format_note(f"{icon} {label} • {ts}{detail_txt}")


def _format_fx_api_status(data: Optional[dict]) -> str:
    if not data:
        return "_Sin llamadas a la API FX._"
    status = data.get("status")
    label = "API FX OK" if status == "success" else "API FX con errores"
    ts = _format_timestamp(data.get("ts"))
    elapsed = data.get("elapsed_ms")
    elapsed_txt = f"{float(elapsed):.0f} ms" if isinstance(elapsed, (int, float)) else "s/d"
    detail = data.get("error")
    detail_txt = f" — {detail}" if detail else ""
    prefix = "✅" if status == "success" else "⚠️"
    return format_note(f"{prefix} {label} • {ts} ({elapsed_txt}){detail_txt}")


def _format_fx_cache_status(data: Optional[dict]) -> str:
    if not data:
        return "_Sin uso de caché registrado._"
    mode = data.get("mode")
    icon = "✅" if mode == "hit" else "ℹ️"
    label = "Uso de caché" if mode == "hit" else "Actualización"
    ts = _format_timestamp(data.get("ts"))
    age = data.get("age")
    age_txt = f"{float(age):.0f}s" if isinstance(age, (int, float)) else "s/d"
    return format_note(f"{icon} {label} • {ts} (edad {age_txt})")


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


def _format_opportunities_status(
    data: Optional[dict],
    history: Optional[Sequence[Mapping[str, Any]]] = None,
    stats: Optional[Mapping[str, Any]] = None,
) -> str:
    if not data and history:
        for entry in reversed(history):
            if isinstance(entry, Mapping):
                data = dict(entry)
                break

    if not data:
        return "_Sin screenings recientes._"

    mode = data.get("mode")
    icon = "✅" if mode == "hit" else "⚙️"
    label = "Cache reutilizada" if mode == "hit" else "Ejecución completa"
    elapsed = data.get("elapsed_ms")
    elapsed_txt = f"{float(elapsed):.0f} ms" if isinstance(elapsed, (int, float)) else "s/d"
    baseline = data.get("cached_elapsed_ms")
    baseline_txt = (
        f" • previo {float(baseline):.0f} ms" if isinstance(baseline, (int, float)) else ""
    )
    ts = _format_timestamp(data.get("ts"))
    metrics_parts: list[str] = []

    def _coerce_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    initial = _coerce_int(data.get("universe_initial"))
    final = _coerce_int(data.get("universe_final"))
    if initial is not None and final is not None:
        metrics_parts.append(f"universo {initial}→{final}")
    elif initial is not None:
        metrics_parts.append(f"universo inicial {initial}")
    elif final is not None:
        metrics_parts.append(f"universo final {final}")

    ratio = _coerce_float(data.get("discard_ratio"))
    if ratio is not None:
        metrics_parts.append(f"descartes {ratio:.0%}")

    sectors_raw = data.get("highlighted_sectors")
    if isinstance(sectors_raw, str):
        sectors = [sectors_raw.strip()] if sectors_raw.strip() else []
    elif isinstance(sectors_raw, Iterable) and not isinstance(
        sectors_raw, (bytes, bytearray, str)
    ):
        sectors = [str(item).strip() for item in sectors_raw if str(item).strip()]
    else:
        sectors = []
    if sectors:
        metrics_parts.append("sectores: " + ", ".join(sectors))

    origin_raw = data.get("counts_by_origin")
    origin_parts: list[str] = []
    if isinstance(origin_raw, Mapping):
        for key, value in origin_raw.items():
            origin_label = str(key).strip()
            if not origin_label:
                continue
            numeric = _coerce_float(value)
            if numeric is None:
                continue
            if float(numeric).is_integer():
                origin_parts.append(f"{origin_label}={int(numeric)}")
            else:
                origin_parts.append(f"{origin_label}={numeric:g}")
    if origin_parts:
        metrics_parts.append("origen: " + ", ".join(origin_parts))

    if stats and isinstance(stats, Mapping):
        summary_parts: list[str] = []
        elapsed_summary = stats.get("elapsed") if isinstance(stats, Mapping) else None
        if isinstance(elapsed_summary, Mapping):
            avg = elapsed_summary.get("avg")
            count = elapsed_summary.get("count")
            stdev = elapsed_summary.get("stdev")
            if isinstance(avg, (int, float)) and isinstance(count, (int, float)):
                count_int = int(count)
                if isinstance(stdev, (int, float)) and stdev > 0:
                    summary_parts.append(
                        f"prom {float(avg):.0f}±{float(stdev):.0f} ms (n={count_int})"
                    )
                else:
                    summary_parts.append(f"prom {float(avg):.0f} ms (n={count_int})")
        hit_ratio = stats.get("hit_ratio") if isinstance(stats, Mapping) else None
        mode_counts = stats.get("mode_counts") if isinstance(stats, Mapping) else None
        total_modes = stats.get("mode_total") if isinstance(stats, Mapping) else None
        if isinstance(hit_ratio, (int, float)) and isinstance(mode_counts, Mapping):
            hit_count = mode_counts.get("hit", 0)
            try:
                hit_count_int = int(hit_count)
            except (TypeError, ValueError):
                hit_count_int = 0
            total = int(total_modes) if isinstance(total_modes, (int, float)) else None
            if total:
                summary_parts.append(
                    f"hits {hit_ratio:.0%} ({hit_count_int}/{total})"
                )
        improvement = stats.get("improvement") if isinstance(stats, Mapping) else None
        if isinstance(improvement, Mapping):
            win_ratio = improvement.get("win_ratio")
            wins = improvement.get("wins")
            count = improvement.get("count")
            delta_avg = improvement.get("avg_delta_ms")
            if isinstance(win_ratio, (int, float)) and isinstance(count, (int, float)):
                try:
                    wins_int = int(wins)
                except (TypeError, ValueError):
                    wins_int = 0
                summary_parts.append(
                    f"mejoras {win_ratio:.0%} ({wins_int}/{int(count)})"
                )
            if isinstance(delta_avg, (int, float)):
                summary_parts.append(f"Δ̄ {float(delta_avg):+0.0f} ms vs caché")
        if summary_parts:
            metrics_parts.append("tendencia: " + " • ".join(summary_parts))

    metrics_txt = f" — {' | '.join(metrics_parts)}" if metrics_parts else ""
    return format_note(
        f"{icon} {label} • {ts} ({elapsed_txt}{baseline_txt}){metrics_txt}"
    )


def _format_opportunities_history(
    history: Optional[Iterable[Mapping[str, Any]]],
    stats: Optional[Mapping[str, Any]] = None,
) -> Iterable[str]:
    if not history:
        return ["_Sin historial disponible._"]

    average_elapsed: Optional[float] = None
    elapsed_stats = None
    if isinstance(stats, Mapping):
        elapsed_stats = stats.get("elapsed")
    if isinstance(elapsed_stats, Mapping):
        avg_candidate = elapsed_stats.get("avg")
        if isinstance(avg_candidate, (int, float)):
            average_elapsed = float(avg_candidate)

    header = "| Momento | Modo | t (ms) | Δ prom | Base |\n| --- | --- | --- | --- | --- |"
    rows = [header]

    for entry in history:
        if not isinstance(entry, Mapping):
            continue
        mode = entry.get("mode")
        icon = "✅" if mode == "hit" else "⚙️"
        label = "hit" if mode == "hit" else "miss"
        ts = _format_timestamp(entry.get("ts"))
        elapsed = entry.get("elapsed_ms")
        elapsed_txt = (
            f"{float(elapsed):.0f}" if isinstance(elapsed, (int, float)) else "s/d"
        )
        baseline = entry.get("cached_elapsed_ms")
        baseline_txt = (
            f"{float(baseline):.0f}" if isinstance(baseline, (int, float)) else "s/d"
        )
        if isinstance(elapsed, (int, float)) and average_elapsed is not None:
            delta = float(elapsed) - average_elapsed
            delta_txt = f"{delta:+.0f}"
        else:
            delta_txt = "s/d"
        rows.append(
            f"| {ts} | {icon} {label} | {elapsed_txt} | {delta_txt} | {baseline_txt} |"
        )

    if len(rows) == 1:
        return ["_Sin historial disponible._"]

    table_block = "\n".join(rows)
    return [table_block]


def render_health_sidebar() -> None:
    """Render the health summary panel inside the sidebar."""
    metrics = get_health_metrics()
    sidebar = st.sidebar
    sidebar.header(f"🩺 Healthcheck (versión {__version__})")
    sidebar.caption("Monitorea la procedencia y el rendimiento de los datos cargados.")

    sidebar.markdown("#### 🔐 Conexión IOL")
    sidebar.markdown(_format_iol_status(metrics.get("iol_refresh")))

    sidebar.markdown("#### 📈 Yahoo Finance")
    sidebar.markdown(_format_yfinance_status(metrics.get("yfinance")))

    sidebar.markdown("#### 💱 FX")
    for line in _format_fx_section(metrics.get("fx_api"), metrics.get("fx_cache")):
        sidebar.markdown(line)

    sidebar.markdown("#### 🔎 Screening de oportunidades")
    sidebar.markdown(
        _format_opportunities_status(
            metrics.get("opportunities"),
            metrics.get("opportunities_history"),
            metrics.get("opportunities_stats"),
        )
    )
    sidebar.markdown("#### 🗂️ Historial de screenings")
    history_entries = metrics.get("opportunities_history") or []
    for line in _format_opportunities_history(
        reversed(history_entries), metrics.get("opportunities_stats")
    ):
        sidebar.markdown(line)

    sidebar.markdown("#### ⏱️ Latencias")
    for line in _format_latency_section(metrics.get("portfolio"), metrics.get("quotes")):
        sidebar.markdown(format_note(line))


__all__ = ["render_health_sidebar"]
