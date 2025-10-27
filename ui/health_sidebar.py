from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import importlib
import logging
from types import ModuleType

import pandas as pd
import streamlit as st

from services.health import get_health_metrics
from shared.time_provider import TimeProvider
from shared.ui import notes as shared_notes
from shared.version import __version__
from ui.actions import render_action_menu
from ui.helpers.navigation import safe_page_link
from ui.sidebar_controls import get_controls_reference_data, render_controls_panel
from ui.ui_settings import render_ui_controls

"""Sidebar panel summarising recent data source health."""


format_note = shared_notes.format_note


logger = logging.getLogger(__name__)

_MONITORING_PANEL_STATE_KEY = "_monitoring_active_panel"
_MONITORING_SHORTCUTS: Sequence[tuple[str, str]] = (
    ("ℹ️ Acerca de", "ui.panels.about"),
    ("🩺 Diagnóstico", "ui.panels.diagnostics"),
    ("🔎 Diagnóstico del sistema", "ui.panels.system_diagnostics"),
    ("🔍 Estado del Sistema", "ui.panels.system_status"),
    ("🔍 IOL RAW", "ui.panels.iol_raw_debug"),
    ("⏱️ Performance", "ui.tabs.performance_dashboard"),
)
_MISSING_PANELS_LOGGED: set[str] = set()
_MONITORING_RENDERED_FLAG = "_monitoring_panel_rendered"


_YFINANCE_PROVIDER_LABELS = {
    "yfinance": "Datos de Yahoo Finance",
    "fallback": "Fallback local",
    "error": "Error o sin datos",
}

_YFINANCE_PROVIDER_SHORT = {
    "yfinance": "YF",
    "fallback": "FB",
    "error": "ERR",
}

_YFINANCE_STATUS_LABELS = {
    "success": "OK",
    "ok": "OK",
    "hit": "OK",
    "fallback": "Fallback",
    "error": "Error",
    "failed": "Error",
    "timeout": "Timeout",
    "missing": "Sin datos",
}

_YFINANCE_ERROR_STATUSES = {"error", "failed", "timeout"}


_TAB_LABELS = {
    "tecnico": "Técnico",
    "riesgo": "Riesgo",
    "fundamentales": "Fundamentales",
}


_RISK_SEVERITY_ICONS = {
    "critical": "🛑",
    "high": "🚨",
    "warning": "⚠️",
    "medium": "⚠️",
    "low": "ℹ️",
}

_RISK_SEVERITY_LABELS = {
    "critical": "Críticas",
    "high": "Altas",
    "warning": "Advertencias",
    "medium": "Medias",
    "low": "Bajas",
}


_GENERIC_STATUS_BADGES = {
    "ok": ("🟢", "Operativo"),
    "success": ("🟢", "Operativo"),
    "healthy": ("🟢", "Operativo"),
    "pass": ("🟢", "Operativo"),
    "warning": ("🟡", "Advertencia"),
    "degraded": ("🟡", "Degradado"),
    "pending": ("🟡", "Pendiente"),
    "info": ("ℹ️", "Informativo"),
    "notice": ("ℹ️", "Informativo"),
    "error": ("🔴", "Error"),
    "failed": ("🔴", "Error"),
    "critical": ("🔴", "Crítico"),
    "timeout": ("🔴", "Timeout"),
}


_FRESHNESS_BADGES = {
    "fresh": ("🟢", "Actualizado"),
    "stale": ("🟡", "Desactualizado"),
    "empty": ("🔴", "Sin datos"),
}

_AUTH_STATUS_BADGES = {
    "authenticated": ("🟢", "Sesión activa"),
    "reauth": ("🟡", "Reautenticación requerida"),
    "reauth_required": ("🟡", "Reautenticación requerida"),
    "expired": ("🟡", "Sesión expirada"),
    "error": ("🔴", "Error de autenticación"),
    "failed": ("🔴", "Error de autenticación"),
    "logout": ("ℹ️", "Sesión finalizada"),
}

_SNAPSHOT_STATUS_BADGES = {
    "hit": ("⚡", "Servido desde snapshot"),
    "miss": ("🆕", "Generado nuevamente"),
    "created": ("💾", "Snapshot creado"),
    "persisted": ("💾", "Snapshot persistido"),
    "error": ("⚠️", "Error de snapshot"),
    "disabled": ("⛔", "Snapshots deshabilitados"),
}


def _get_session_profile() -> Optional[Mapping[str, Any]]:
    profile = st.session_state.get("iol_user_profile")
    return profile if isinstance(profile, Mapping) else None


def _format_profile_preferences(preferencias: Any) -> str:
    if isinstance(preferencias, (list, tuple)):
        items = [str(item).strip() for item in preferencias if str(item).strip()]
        if items:
            return ", ".join(items)
    return "No disponible"


def _render_investor_profile_section(host: Any) -> None:
    profile = _get_session_profile()
    section = host.container(border=True)
    with section:
        st.markdown("#### 👤 Perfil del inversor")
        if st.button("🔄 Actualizar perfil"):
            st.cache_data.clear()
            st.session_state["_profile_refresh_pending"] = True
            st.session_state.pop("iol_user_profile", None)
        if profile:
            nombre = profile.get("nombre") or "No disponible"
            perfil_inversor = profile.get("perfil_inversor") or "No disponible"
            preferencias = _format_profile_preferences(profile.get("preferencias"))
            st.markdown(f"👤 **Nombre:** {nombre}")
            st.markdown(f"📊 **Perfil inversor:** {perfil_inversor}")
            st.markdown(f"💡 **Preferencias:** {preferencias}")
        else:
            st.markdown("_Perfil del inversor no disponible._")


def _sanitize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_status_key(value: Any) -> str:
    text = _sanitize_text(value)
    return text.casefold() if text else ""


def _coerce_timestamp(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _freshness_badge(value: Any) -> tuple[str, str]:
    key = _normalize_status_key(value)
    if not key:
        return "⚪️", "Estado desconocido"
    icon, label = _FRESHNESS_BADGES.get(key, ("⚪️", "Estado desconocido"))
    return icon, label


def _auth_status_badge(value: Any) -> tuple[str, str]:
    key = _normalize_status_key(value)
    icon, label = _AUTH_STATUS_BADGES.get(key, ("ℹ️", "Estado desconocido"))
    return icon, label


def _snapshot_status_badge(value: Any) -> tuple[str, str]:
    key = _normalize_status_key(value)
    icon, label = _SNAPSHOT_STATUS_BADGES.get(key, ("🗂️", "Estado desconocido"))
    return icon, label


def _risk_severity_badge(value: Any) -> tuple[str, str]:
    key = _normalize_status_key(value)
    icon = _RISK_SEVERITY_ICONS.get(key, "ℹ️")
    label = _RISK_SEVERITY_LABELS.get(key)
    if not label:
        raw_text = _sanitize_text(value)
        label = raw_text.title() if raw_text else "Desconocido"
    return icon, label


def _status_badge(value: Any, *, default_label: str = "Estado desconocido") -> tuple[str, str]:
    key = _normalize_status_key(value)
    icon, label = _GENERIC_STATUS_BADGES.get(key, ("ℹ️", default_label))
    return icon, label


_STATUS_SEVERITY_SUCCESS = {"ok", "success", "healthy", "pass"}
_STATUS_SEVERITY_WARNING = {"warning", "degraded", "pending", "info", "notice"}
_STATUS_SEVERITY_DANGER = {
    "error",
    "failed",
    "critical",
    "timeout",
    "missing",
}


def _categorize_status(value: Any) -> str:
    key = _normalize_status_key(value)
    if key in _STATUS_SEVERITY_SUCCESS:
        return "success"
    if key in _STATUS_SEVERITY_WARNING:
        return "warning"
    if key in _STATUS_SEVERITY_DANGER:
        return "danger"
    return "unknown"


def _format_duration_seconds(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None

    try:
        value = float(seconds)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None

    if value >= 3600:
        hours = value / 3600
        return f"{hours:.1f} h"
    if value >= 120:
        minutes = value / 60
        return f"{minutes:.1f} min"
    if value >= 1:
        return f"{value:.2f} s"
    return f"{value * 1000:.0f} ms"


def _format_session_monitoring(data: Optional[Mapping[str, Any]]) -> Iterable[str]:
    if not isinstance(data, Mapping):
        return ["_Sin métricas de sesiones._"]

    lines: list[str] = []
    status_icon, status_label = _status_badge(data.get("status"))
    ts_text = _format_timestamp(_coerce_timestamp(data.get("ts")))
    detail_text = _sanitize_text(data.get("detail") or data.get("message"))

    summary_parts = [f"{status_icon} Estado {status_label.lower()}"]
    if ts_text:
        summary_parts.append(ts_text)
    summary = " • ".join(summary_parts)
    if detail_text:
        summary += f" — {detail_text}"
    lines.append(format_note(summary))

    active_sessions = data.get("active_sessions")
    active_summary = _format_active_sessions(active_sessions)
    if active_summary:
        lines.append(format_note(f"👥 Sesiones activas: {active_summary}"))

    avg_login = _format_avg_login_to_render(data.get("avg_login_to_render"))
    if avg_login:
        lines.append(format_note(f"⏱️ Promedio login→render: {avg_login}"))

    http_block = data.get("http_errors")
    if isinstance(http_block, Mapping):
        count_value = http_block.get("count")
        summary_bits: list[str] = []
        if isinstance(count_value, (int, float)) and int(count_value) > 0:
            summary_bits.append(f"🚨 Errores HTTP {int(count_value)}")
        last_entry = http_block.get("last")
        last_ts = None
        if isinstance(last_entry, Mapping):
            last_ts = _coerce_timestamp(last_entry.get("ts"))
        if last_ts is not None:
            summary_bits.append(f"último {_format_timestamp(last_ts)}")
        if summary_bits:
            lines.append(format_note(" • ".join(summary_bits)))
    else:
        last_error = _format_http_error(data.get("last_http_error"))
        if last_error:
            lines.append(last_error)

    if len(lines) == 1:
        lines.append("_Sin datos de sesiones adicionales._")

    return lines


def _normalize_numeric_samples(data: Any) -> list[float]:
    if isinstance(data, Mapping):
        raw_samples = data.get("samples")
    else:
        raw_samples = data

    if not isinstance(raw_samples, Iterable) or isinstance(raw_samples, (str, bytes, bytearray)):
        return []

    samples: list[float] = []
    for entry in raw_samples:
        try:
            numeric = float(entry)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        samples.append(float(numeric))
    return samples


def _extract_stat_samples(entry: Any, metric: str) -> list[float]:
    if not isinstance(entry, Mapping):
        return []

    stats = entry.get("stats")
    if not isinstance(stats, Mapping):
        return []

    block = stats.get(metric)
    if not isinstance(block, Mapping):
        return []

    return _normalize_numeric_samples(block.get("samples"))


def _build_series_dataframe(
    series: Mapping[str, Iterable[float]],
) -> Optional[pd.DataFrame]:
    columns: dict[str, pd.Series] = {}
    for label, values in series.items():
        numeric_values = list(values)
        if not numeric_values:
            continue
        columns[label] = pd.Series(numeric_values, dtype="float64")

    if not columns:
        return None

    frame = pd.DataFrame(columns)
    frame.index = frame.index + 1
    frame.index.name = "Muestra"
    return frame


def _collect_latency_series(metrics: Mapping[str, Any]) -> Mapping[str, list[float]]:
    series: dict[str, list[float]] = {}
    entries = [
        ("portfolio", "Portafolio (ms)"),
        ("quotes", "Cotizaciones (ms)"),
        ("fx_api", "FX API (ms)"),
    ]
    for key, label in entries:
        samples = _extract_stat_samples(metrics.get(key), "latency")
        if samples:
            series[label] = samples
    return series


def _extract_semver(text: str) -> Optional[str]:
    match = re.search(r"\d+(?:\.\d+){0,2}", text)
    if not match:
        return None
    version = match.group(0)
    parts = version.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return version


def _format_version_token(value: Any, *, icon: str, label: Optional[str] = None) -> Optional[str]:
    text = _sanitize_text(value)
    if not text:
        return None
    version = _extract_semver(text) or text
    if label:
        return f"{icon} {label} {version}"
    return f"{icon} {version}"


def _format_environment_badge(snapshot: Any) -> Optional[str]:
    if not isinstance(snapshot, Mapping):
        return None

    python_token = _format_version_token(snapshot.get("python_version") or snapshot.get("python"), icon="🐍")
    streamlit_token = _format_version_token(
        snapshot.get("streamlit_version") or snapshot.get("streamlit"),
        icon="📊",
        label="Streamlit",
    )
    runtime_label = _sanitize_text(snapshot.get("runtime") or snapshot.get("environment") or snapshot.get("platform"))
    runtime_token = f"🖥️ {runtime_label}" if runtime_label else None

    tokens = [token for token in (python_token, streamlit_token, runtime_token) if token]
    if not tokens:
        return None

    ts_value = _coerce_timestamp(snapshot.get("ts"))
    ts_text = _format_timestamp(ts_value)
    if ts_text and ts_text != "Sin registro":
        tokens.append(ts_text)

    return format_note(" • ".join(tokens))


def _format_recent_http_error(metrics: Mapping[str, Any]) -> Optional[str]:
    monitoring = metrics.get("session_monitoring")
    if isinstance(monitoring, Mapping):
        http_block = monitoring.get("http_errors")
        if isinstance(http_block, Mapping):
            formatted = _format_http_error(http_block.get("last"))
            if formatted:
                return formatted
            count_value = http_block.get("count")
            if isinstance(count_value, (int, float)) and int(count_value) > 0:
                return format_note(f"🚨 Errores HTTP registrados: {int(count_value)}")

    raw_error = metrics.get("last_http_error")
    if raw_error:
        formatted = _format_http_error(raw_error)
        if formatted:
            return formatted
    return None


def _mirror_sidebar_render(host: Any, method: str, *args: Any, **kwargs: Any) -> None:
    sidebar_host = getattr(host, "_host", None)
    if sidebar_host is None:
        return
    renderer = getattr(sidebar_host, method, None)
    if not callable(renderer):
        return
    try:
        renderer(*args, **kwargs)
    except TypeError:
        renderer(*args)


def _render_chart(host: Any, method: str, *args: Any, **kwargs: Any) -> bool:
    renderer = getattr(host, method, None)
    if callable(renderer):
        renderer(*args, **kwargs)
        _mirror_sidebar_render(host, method, *args, **kwargs)
        return True

    fallback = getattr(st, method, None)
    if callable(fallback):
        fallback(*args, **kwargs)
        return True

    return False


def _render_recent_stats(host: Any, metrics: Mapping[str, Any]) -> None:
    st.markdown("#### 📈 Estadísticas recientes")

    charts_rendered = False

    latency_series = _collect_latency_series(metrics)
    latency_frame = _build_series_dataframe(latency_series)
    if latency_frame is not None:
        charts_rendered |= _render_chart(host, "line_chart", latency_frame)

    cache_samples = _extract_stat_samples(metrics.get("fx_cache"), "age")
    if cache_samples:
        cache_frame = pd.DataFrame({"Edad caché (s)": pd.Series(cache_samples, dtype="float64")})
        cache_frame.index = cache_frame.index + 1
        cache_frame.index.name = "Muestra"
        charts_rendered |= _render_chart(host, "area_chart", cache_frame)

    http_note = _format_recent_http_error(metrics)
    if http_note:
        host.markdown(http_note)
        charts_rendered = True

    if not charts_rendered:
        host.markdown("_Sin estadísticas recientes._")


def _format_active_sessions(data: Any) -> Optional[str]:
    if isinstance(data, Mapping):
        parts: list[str] = []
        current = data.get("current") or data.get("value")
        if isinstance(current, (int, float)):
            parts.append(f"actual {int(current)}")
        peak = data.get("peak") or data.get("max")
        if isinstance(peak, (int, float)):
            parts.append(f"máximo {int(peak)}")
        window = _sanitize_text(data.get("window"))
        if window:
            parts.append(f"ventana {window}")
        return " • ".join(parts) if parts else None
    if isinstance(data, (int, float)):
        return str(int(data))
    text = _sanitize_text(data)
    return text


def _format_avg_login_to_render(data: Any) -> Optional[str]:
    if isinstance(data, Mapping):
        seconds_value: Optional[float] = None
        for key in ("seconds", "secs", "sec", "value", "avg", "mean"):
            candidate = data.get(key)
            if isinstance(candidate, (int, float)):
                seconds_value = float(candidate)
                break
        if seconds_value is None:
            for key in ("milliseconds", "ms"):
                candidate = data.get(key)
                if isinstance(candidate, (int, float)):
                    seconds_value = float(candidate) / 1000
                    break
        formatted = _format_duration_seconds(seconds_value)
        if not formatted:
            return None
        samples = data.get("samples") or data.get("count")
        if isinstance(samples, (int, float)) and int(samples) > 0:
            formatted += f" (n={int(samples)})"
        return formatted
    if isinstance(data, (int, float)):
        seconds_value = float(data)
        if seconds_value >= 1000:
            seconds_value = seconds_value / 1000
        return _format_duration_seconds(seconds_value)
    text = _sanitize_text(data)
    return text


def _format_http_error(data: Any) -> Optional[str]:
    if not isinstance(data, Mapping):
        return None

    status_code = data.get("status") or data.get("status_code")
    method = _sanitize_text(data.get("method"))
    path = _sanitize_text(data.get("path") or data.get("endpoint") or data.get("url"))
    ts_text = _format_timestamp(_coerce_timestamp(data.get("ts")))
    detail_text = _sanitize_text(data.get("detail") or data.get("message"))

    header = "🚨 Último error HTTP"
    if status_code is not None:
        header += f" {status_code}"
    tokens = [header]
    if method:
        tokens.append(method.upper())
    if path:
        tokens.append(path)
    if ts_text:
        tokens.append(ts_text)
    summary = " • ".join(tokens)
    if detail_text:
        summary += f" — {detail_text}"
    return format_note(summary)


def _iter_diagnostic_checks(
    checks: Any,
) -> Iterable[tuple[Optional[str], Mapping[str, Any]]]:
    if isinstance(checks, Mapping):
        for key, value in checks.items():
            if isinstance(value, Mapping):
                yield str(key), value
    elif isinstance(checks, Iterable) and not isinstance(checks, (str, bytes, bytearray)):
        for entry in checks:
            if isinstance(entry, Mapping):
                yield None, entry


def _format_diagnostic_entry(name: Optional[str], data: Mapping[str, Any]) -> Optional[str]:
    status_icon, status_label = _status_badge(data.get("status"))
    label = _sanitize_text(data.get("label") or data.get("name")) or (str(name).strip() if name else None) or "Chequeo"
    ts_text = _format_timestamp(_coerce_timestamp(data.get("ts")))
    detail_text = _sanitize_text(data.get("detail") or data.get("message"))

    summary_parts = [f"{status_icon} {label}", status_label.lower()]
    if ts_text:
        summary_parts.append(ts_text)
    summary = " • ".join(summary_parts)
    if detail_text:
        summary += f" — {detail_text}"
    return format_note(summary)


def _format_diagnostics_section(data: Optional[Mapping[str, Any]]) -> Iterable[str]:
    if not isinstance(data, Mapping):
        return ["_Sin diagnósticos registrados._"]

    group = data.get("initial")
    if not isinstance(group, Mapping):
        group = data

    lines: list[str] = []
    status_icon, status_label = _status_badge(group.get("status"))
    ts_text = _format_timestamp(_coerce_timestamp(group.get("ts")))
    detail_text = _sanitize_text(group.get("detail") or group.get("message"))

    summary_parts = [f"{status_icon} Diagnóstico {status_label.lower()}"]
    if ts_text:
        summary_parts.append(ts_text)
    summary = " • ".join(summary_parts)
    if detail_text:
        summary += f" — {detail_text}"
    lines.append(format_note(summary))

    for name, entry in _iter_diagnostic_checks(group.get("checks") or group.get("results")):
        formatted = _format_diagnostic_entry(name, entry)
        if formatted:
            lines.append(formatted)

    if len(lines) == 1:
        lines.append("_Sin chequeos registrados._")

    return lines


def _format_dependencies_section(data: Optional[Mapping[str, Any]]) -> Iterable[str]:
    if not isinstance(data, Mapping):
        return ["_Sin registros de dependencias._"]

    items = data.get("items")
    if not isinstance(items, Mapping) or not items:
        return ["_Sin registros de dependencias._"]

    lines: list[str] = []

    overall_status = data.get("status")
    if isinstance(overall_status, str) and overall_status.strip():
        icon, label = _status_badge(overall_status)
        summary = f"{icon} Dependencias {label.lower()}"
        lines.append(format_note(summary))

    for name in sorted(items):
        entry = items[name]
        if not isinstance(entry, Mapping):
            continue
        icon, status_label = _status_badge(entry.get("status"))
        label_text = _sanitize_text(entry.get("label")) or _sanitize_text(name) or "Dependencia"
        ts_text = _format_timestamp(_coerce_timestamp(entry.get("ts")))
        detail_text = _sanitize_text(entry.get("detail"))

        summary_parts = [f"{icon} {label_text}", status_label.lower()]
        if ts_text:
            summary_parts.append(ts_text)
        summary = " • ".join(summary_parts)
        if detail_text:
            summary += f" — {detail_text}"
        lines.append(format_note(summary))

    return lines or ["_Sin registros de dependencias._"]


def _format_risk_summary(data: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(data, Mapping):
        return "_Sin incidencias de riesgo registradas._"

    total = data.get("total")
    if not isinstance(total, (int, float)) or int(total) <= 0:
        return "_Sin incidencias de riesgo registradas._"

    total_int = int(total)
    severity_stats = data.get("by_severity") if isinstance(data.get("by_severity"), Mapping) else {}
    severity_tokens: list[str] = []
    if isinstance(severity_stats, Mapping):
        for key in sorted(severity_stats):
            stats = severity_stats.get(key)
            if not isinstance(stats, Mapping):
                continue
            count_val = stats.get("count")
            if not isinstance(count_val, (int, float)):
                continue
            count_int = int(count_val)
            if count_int <= 0:
                continue
            icon, label = _risk_severity_badge(key)
            severity_tokens.append(f"{icon} {label} {count_int}")

    fallback_count = data.get("fallback_count")
    fallback_ratio = data.get("fallback_ratio")
    if isinstance(fallback_count, (int, float)) and total_int:
        fallback_int = int(fallback_count)
        if fallback_int:
            if isinstance(fallback_ratio, (int, float)):
                severity_tokens.append(f"🛟 Fallbacks {fallback_int}/{total_int} ({fallback_ratio:.0%})")
            else:
                severity_tokens.append(f"🛟 Fallbacks {fallback_int}/{total_int}")

    tail = f" — {' • '.join(severity_tokens)}" if severity_tokens else ""
    return format_note(f"🚨 Incidencias {total_int}{tail}")


def _format_risk_latest(entry: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not isinstance(entry, Mapping):
        return None

    category = _sanitize_text(entry.get("category")) or "desconocido"
    icon, severity_label = _risk_severity_badge(entry.get("severity"))

    ts_value = entry.get("ts")
    ts_text: Optional[str] = None
    if isinstance(ts_value, (int, float)):
        ts_text = _format_timestamp(float(ts_value))
    elif isinstance(ts_value, str):
        try:
            ts_text = _format_timestamp(float(ts_value))
        except ValueError:
            ts_text = None
    if not ts_text:
        ts_text = "Sin registro"

    detail = _sanitize_text(entry.get("detail"))
    detail_suffix = f" — {detail}" if detail else ""

    fallback_flag = entry.get("fallback")
    fallback_suffix = " • 🛟 con fallback" if bool(fallback_flag) else ""

    source_text = _sanitize_text(entry.get("source"))
    source_suffix = f" • origen: {source_text}" if source_text else ""

    tags = entry.get("tags")
    tags_suffix = ""
    if isinstance(tags, Iterable) and not isinstance(tags, (str, bytes, bytearray)):
        tag_tokens = [str(tag).strip() for tag in tags if str(tag).strip()]
        if tag_tokens:
            tags_suffix = f" • tags: {', '.join(tag_tokens)}"

    return format_note(
        f"{icon} Última incidencia en {category}: {severity_label} • {ts_text}"
        f"{fallback_suffix}{source_suffix}{tags_suffix}{detail_suffix}"
    )


def _format_risk_detail_line(category: str, stats: Mapping[str, Any]) -> str:
    label = _sanitize_text(stats.get("label")) or category or "desconocido"
    count_val = stats.get("count")
    count_int = int(count_val) if isinstance(count_val, (int, float)) else 0

    severity_counts = stats.get("severity_counts") if isinstance(stats.get("severity_counts"), Mapping) else {}
    severity_tokens: list[str] = []
    if isinstance(severity_counts, Mapping):
        for severity_key in sorted(severity_counts):
            raw_value = severity_counts.get(severity_key)
            if not isinstance(raw_value, (int, float)):
                continue
            value_int = int(raw_value)
            if value_int <= 0:
                continue
            icon, severity_label = _risk_severity_badge(severity_key)
            severity_tokens.append(f"{icon} {severity_label} {value_int}")

    fallback_token: Optional[str] = None
    fallback_count = stats.get("fallback_count")
    fallback_ratio = stats.get("fallback_ratio")
    if isinstance(fallback_count, (int, float)) and count_int:
        fallback_int = int(fallback_count)
        if fallback_int:
            if isinstance(fallback_ratio, (int, float)):
                fallback_token = f"Fallbacks 🛟 {fallback_int}/{count_int} ({fallback_ratio:.0%})"
            else:
                fallback_token = f"Fallbacks 🛟 {fallback_int}/{count_int}"
    metrics_parts = [token for token in severity_tokens if token]
    if fallback_token:
        metrics_parts.append(fallback_token)
    metrics_summary = " • ".join(metrics_parts) if metrics_parts else "sin desglose"

    ts_value = stats.get("last_ts")
    ts_text = None
    if isinstance(ts_value, (int, float)):
        ts_text = _format_timestamp(float(ts_value))

    last_severity = stats.get("last_severity")
    last_detail = _sanitize_text(stats.get("last_detail"))
    last_fallback = stats.get("last_fallback")
    last_source = _sanitize_text(stats.get("last_source"))
    last_tags = stats.get("last_tags")

    status_parts: list[str] = []
    if last_severity:
        icon, severity_label = _risk_severity_badge(last_severity)
        status_parts.append(f"Último: {icon} {severity_label}")
    if last_fallback:
        status_parts.append("Fallback reciente")
    if last_source:
        status_parts.append(f"origen: {last_source}")
    if ts_text:
        status_parts.append(ts_text)
    if isinstance(last_tags, Iterable) and not isinstance(last_tags, (str, bytes, bytearray)):
        tag_tokens = [str(tag).strip() for tag in last_tags if str(tag).strip()]
        if tag_tokens:
            status_parts.append("tags: " + ", ".join(tag_tokens))
    if last_detail:
        status_parts.append(last_detail)

    detail_suffix = f" • {' | '.join(status_parts)}" if status_parts else ""

    return format_note(f"🧮 {label} — {count_int} incidencias • {metrics_summary}{detail_suffix}")


def _format_risk_section(data: Optional[Mapping[str, Any]]) -> Iterable[str]:
    summary_line = _format_risk_summary(data)
    if summary_line.startswith("_Sin incidencias"):
        return [summary_line]

    lines = [summary_line]
    if isinstance(data, Mapping):
        latest_line = _format_risk_latest(data.get("latest"))
        if latest_line:
            lines.append(latest_line)
    return lines


def _format_risk_detail_section(data: Optional[Mapping[str, Any]]) -> Iterable[str]:
    if not isinstance(data, Mapping):
        return ["_Sin incidencias registradas._"]

    categories = data.get("by_category")
    if not isinstance(categories, Mapping) or not categories:
        return ["_Sin incidencias registradas._"]

    lines: list[str] = []
    for key in sorted(categories):
        stats = categories.get(key)
        if not isinstance(stats, Mapping):
            continue
        lines.append(_format_risk_detail_line(str(key), stats))

    return lines or ["_Sin incidencias registradas._"]


def _extract_yfinance_history(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    history_raw = data.get("history")
    entries: list[Mapping[str, Any]] = []
    if isinstance(history_raw, Iterable) and not isinstance(history_raw, (str, bytes, bytearray)):
        for entry in history_raw:
            if isinstance(entry, Mapping):
                entries.append(entry)
    return entries


def _compress_yfinance_history(entries: Sequence[Mapping[str, Any]], *, limit: int = 5) -> Optional[str]:
    if not entries:
        return None

    recent = entries[-limit:]
    tokens: list[str] = []
    for entry in recent:
        provider_value = _sanitize_text(entry.get("provider")) or "desconocido"
        provider_key = provider_value.casefold()
        label = _YFINANCE_PROVIDER_SHORT.get(provider_key, provider_value[:3].upper())
        status_key = _normalize_status_key(entry.get("result") or entry.get("status"))
        fallback_flag = bool(entry.get("fallback"))
        if status_key in _YFINANCE_ERROR_STATUSES:
            icon = "⚠️"
        elif fallback_flag:
            icon = "🛟"
        else:
            icon = "✅"
        tokens.append(f"{icon}{label}")

    if not tokens:
        return None

    return "Historial: " + " · ".join(tokens)


def _format_timestamp(ts: Optional[float]) -> str:
    if not ts:
        return "Sin registro"
    snapshot = TimeProvider.from_timestamp(ts)
    if snapshot is None:
        return "Sin registro"
    return snapshot.text


def _format_authentication_section(data: Optional[Mapping[str, Any]]) -> Iterable[str]:
    if not isinstance(data, Mapping):
        return ["_Sin autenticaciones registradas._"]

    status_icon, status_label = _auth_status_badge(data.get("status"))
    freshness_value = data.get("freshness")
    freshness_icon, freshness_label = _freshness_badge(freshness_value)

    user_text = _sanitize_text(data.get("user")) or "desconocido"
    method_text = _sanitize_text(data.get("method"))
    scope_text = _sanitize_text(data.get("scope"))
    detail_text = _sanitize_text(data.get("detail") or data.get("message"))
    error_text = _sanitize_text(data.get("error"))

    last_ok_ts = _coerce_timestamp(data.get("last_success_ts") or data.get("last_refresh_ts") or data.get("ts"))
    last_error_ts = _coerce_timestamp(data.get("last_error_ts") or data.get("last_failure_ts"))

    token_age = data.get("token_age_secs") or data.get("token_age")
    token_age_txt = f"token {float(token_age):.0f}s" if isinstance(token_age, (int, float)) else None

    expires_in = data.get("expires_in_secs") or data.get("expires_in")
    expires_txt = f"expira en {float(expires_in):.0f}s" if isinstance(expires_in, (int, float)) else None

    reauth_flag = bool(data.get("reauth_required") or data.get("reauth"))

    parts = [f"{freshness_icon} {status_icon} {status_label}"]
    parts.append(f"usuario: {user_text}")
    if freshness_value is not None:
        parts.append(f"estado {freshness_label.lower()}")
    ok_ts_text = _format_timestamp(last_ok_ts)
    parts.append(f"último ok {ok_ts_text}")
    if last_error_ts is not None:
        error_ts_text = _format_timestamp(last_error_ts)
        parts.append(f"último error {error_ts_text}")
    if method_text:
        parts.append(f"método {method_text}")
    if scope_text:
        parts.append(f"scope {scope_text}")
    if token_age_txt:
        parts.append(token_age_txt)
    if expires_txt:
        parts.append(expires_txt)
    if reauth_flag:
        parts.append("requiere login")

    suffix_bits: list[str] = []
    if detail_text:
        suffix_bits.append(detail_text)
    if error_text:
        suffix_bits.append(error_text)
    suffix = f" — {' • '.join(suffix_bits)}" if suffix_bits else ""

    return [format_note(" • ".join(parts) + suffix)]


def _format_snapshot_section(data: Optional[Mapping[str, Any]]) -> Iterable[str]:
    if not isinstance(data, Mapping):
        return ["_Sin snapshots registrados._"]

    status_icon, status_label = _snapshot_status_badge(data.get("status"))
    freshness_value = data.get("freshness")
    freshness_icon, freshness_label = _freshness_badge(freshness_value)

    last_use_ts = _coerce_timestamp(data.get("last_used_ts") or data.get("last_load_ts") or data.get("last_access_ts"))
    last_saved_ts = _coerce_timestamp(data.get("last_saved_ts") or data.get("last_created_ts") or data.get("ts"))

    backend = data.get("backend")
    backend_label = None
    backend_path = None
    if isinstance(backend, Mapping):
        backend_label = _sanitize_text(backend.get("label") or backend.get("kind"))
        backend_path = _sanitize_text(backend.get("path"))
    else:
        backend_label = _sanitize_text(backend)

    snapshot_id = _sanitize_text(data.get("snapshot_id") or data.get("last_snapshot_id"))
    comparison_id = _sanitize_text(data.get("comparison_id"))

    hits = data.get("hits") or data.get("hit_count")
    misses = data.get("misses") or data.get("miss_count")
    reuse_count = data.get("reuse_count")

    parts = [f"{freshness_icon} {status_icon} {status_label}"]
    if freshness_value is not None:
        parts.append(f"estado {freshness_label.lower()}")
    if last_use_ts is not None:
        parts.append(f"último uso {_format_timestamp(last_use_ts)}")
    if last_saved_ts is not None:
        parts.append(f"último guardado {_format_timestamp(last_saved_ts)}")
    if backend_label:
        parts.append(f"backend {backend_label}")
    if isinstance(hits, (int, float)):
        parts.append(f"hits {int(hits)}")
    if isinstance(misses, (int, float)):
        parts.append(f"misses {int(misses)}")
    if isinstance(reuse_count, (int, float)):
        parts.append(f"reusos {int(reuse_count)}")

    detail_text = _sanitize_text(data.get("detail") or data.get("message"))
    suffix = f" — {detail_text}" if detail_text else ""

    lines = [format_note(" • ".join(parts) + suffix)]

    path_bits: list[str] = []
    if backend_path:
        path_bits.append(f"ruta: {backend_path}")
    storage_count = data.get("stored_snapshots") or data.get("snapshot_count")
    if isinstance(storage_count, (int, float)):
        path_bits.append(f"snapshots {int(storage_count)}")
    if snapshot_id:
        path_bits.append(f"último id {snapshot_id}")
    if comparison_id:
        path_bits.append(f"comparación {comparison_id}")

    if path_bits:
        lines.append(format_note("📁 " + " • ".join(path_bits)))

    return lines


def _format_iol_status(data: Optional[dict]) -> str:
    if not data:
        return "_Sin actividad registrada._"
    status = data.get("status")
    label = "Refresh correcto" if status == "success" else "Error al refrescar"
    ts_value = _coerce_timestamp(data.get("last_fetch_ts") or data.get("ts"))
    ts = _format_timestamp(ts_value)
    detail = data.get("detail")
    detail_txt = f" — {detail}" if detail else ""
    prefix = "✅" if status == "success" else "⚠️"
    freshness_value = data.get("freshness")
    freshness_icon, freshness_label = _freshness_badge(freshness_value)
    parts = [f"{freshness_icon} {prefix} {label}"]
    parts.append(f"último fetch {ts}")
    if freshness_value is not None:
        parts.append(f"estado {freshness_label.lower()}")
    return format_note(" • ".join(parts) + detail_txt)


def _format_yfinance_status(data: Optional[dict]) -> str:
    if not data:
        return "_Sin consultas registradas._"
    history_entries = _extract_yfinance_history(data)
    latest_entry: Mapping[str, Any] | dict[str, Any]
    if history_entries:
        latest_entry = history_entries[-1]
    else:
        latest_entry = data

    provider_value = (
        _sanitize_text(latest_entry.get("provider"))
        or _sanitize_text(data.get("latest_provider"))
        or _sanitize_text(data.get("source"))
        or "desconocido"
    )
    provider_key = provider_value.casefold()
    provider_label = _YFINANCE_PROVIDER_LABELS.get(provider_key, f"Fuente: {provider_value}")

    fallback_value = latest_entry.get("fallback")
    if fallback_value is None:
        fallback_value = data.get("fallback")
    if fallback_value is None:
        fallback_flag = provider_key != "yfinance"
    else:
        fallback_flag = bool(fallback_value)

    result_raw = (
        latest_entry.get("result")
        or latest_entry.get("status")
        or data.get("latest_result")
        or data.get("result")
        or data.get("status")
    )
    result_key = _normalize_status_key(result_raw)
    if not result_key:
        result_key = "fallback" if fallback_flag else "success"

    if result_key in _YFINANCE_ERROR_STATUSES:
        icon = "⚠️"
    elif fallback_flag:
        icon = "🛟"
    else:
        icon = "✅"

    status_label = _YFINANCE_STATUS_LABELS.get(result_key, result_key.title() if result_key else "Desconocido")

    timestamp_value = None
    if isinstance(latest_entry, Mapping):
        timestamp_value = _coerce_timestamp(latest_entry.get("last_fetch_ts") or latest_entry.get("ts"))
    if timestamp_value is None:
        timestamp_value = _coerce_timestamp(data.get("last_fetch_ts") or data.get("ts"))
    ts_text = _format_timestamp(timestamp_value)

    detail_value = None
    if isinstance(latest_entry, Mapping):
        detail_value = latest_entry.get("detail")
    if detail_value is None:
        detail_value = data.get("detail")
    detail_text = _sanitize_text(detail_value)
    detail_suffix = f" — {detail_text}" if detail_text else ""

    fallback_badge = " [Fallback]" if fallback_flag else ""
    history_summary = _compress_yfinance_history(history_entries)

    freshness_value = data.get("freshness") or latest_entry.get("freshness")
    freshness_icon, freshness_label = _freshness_badge(freshness_value)

    parts = [
        f"{freshness_icon} {icon} {provider_label}{fallback_badge}",
        f"• último fetch {ts_text}",
        f"• Resultado: {status_label}{detail_suffix}",
    ]
    if freshness_value is not None:
        parts.append(f"• estado {freshness_label.lower()}")
    if history_summary:
        parts.append(f"• {history_summary}")

    return format_note(" ".join(parts))


def _format_rate_limit_reason(reason: Optional[str]) -> Optional[str]:
    if not reason:
        return None
    text = str(reason).strip()
    if not text:
        return None
    normalized = text.casefold()
    mapping = {
        "throttle": "pre-llamada",
        "http_429": "HTTP 429",
    }
    return mapping.get(normalized, text)


def _format_quote_providers(data: Optional[Mapping[str, Any]]) -> Iterable[str]:
    if not isinstance(data, Mapping):
        return ["_Sin consultas de cotizaciones registradas._"]

    providers = data.get("providers")
    if not isinstance(providers, Iterable) or isinstance(providers, (str, bytes, bytearray)):
        return ["_Sin consultas de cotizaciones registradas._"]

    entries: list[Mapping[str, Any]] = [entry for entry in providers if isinstance(entry, Mapping)]
    if not entries:
        return ["_Sin consultas de cotizaciones registradas._"]

    total_val = data.get("total")
    total_int = int(total_val) if isinstance(total_val, (int, float)) else None
    ok_val = data.get("ok_total")
    ok_int = int(ok_val) if isinstance(ok_val, (int, float)) else None
    ok_ratio = data.get("ok_ratio") if isinstance(data.get("ok_ratio"), (int, float)) else None
    stale_total_val = data.get("stale_total")
    stale_total = int(stale_total_val) if isinstance(stale_total_val, (int, float)) else 0
    rate_total_val = data.get("rate_limit_total")
    rate_total = int(rate_total_val) if isinstance(rate_total_val, (int, float)) else 0

    header_parts: list[str] = []
    if total_int is not None:
        header_parts.append(f"Total {total_int}")
    if ok_int is not None and total_int:
        if ok_ratio is not None:
            header_parts.append(f"OK {ok_int}/{total_int} ({ok_ratio:.0%})")
        else:
            header_parts.append(f"OK {ok_int}/{total_int}")
    if stale_total:
        header_parts.append(f"Stale {stale_total}")
    if rate_total:
        header_parts.append(f"Rate limit {rate_total}")

    lines: list[str] = []
    if header_parts:
        lines.append(format_note("📊 " + " • ".join(header_parts)))

    http_counters_val = data.get("http_counters") if isinstance(data.get("http_counters"), Mapping) else None
    http_counters: Mapping[str, Any] = http_counters_val or {}

    if http_counters:
        iol_500 = http_counters.get("iolv2_500", 0)
        legacy_429 = http_counters.get("legacy_429", 0)
        legacy_auth = http_counters.get("legacy_auth_fail", 0)
        if iol_500:
            lines.append(format_note(f"⚠️ IOL v2 HTTP 500: {iol_500}"))
        if legacy_429:
            lines.append(format_note(f"🚦 Legacy rate-limit 429: {legacy_429}"))
        if legacy_auth:
            lines.append(format_note(f"🔐 Legacy auth fallida: {legacy_auth}"))

    for entry in entries:
        label = _sanitize_text(entry.get("label")) or str(entry.get("provider") or "desconocido")
        count_val = entry.get("count")
        count_int = int(count_val) if isinstance(count_val, (int, float)) else 0
        stale_count_val = entry.get("stale_count")
        stale_count = int(stale_count_val) if isinstance(stale_count_val, (int, float)) else 0
        ok_count_val = entry.get("ok_count")
        ok_count = int(ok_count_val) if isinstance(ok_count_val, (int, float)) else 0
        ok_ratio_val = entry.get("ok_ratio")
        ok_ratio = float(ok_ratio_val) if isinstance(ok_ratio_val, (int, float)) else None
        avg_ms = entry.get("avg_ms")
        last_ms = entry.get("last_ms")
        p50_ms = entry.get("p50_ms")
        p95_ms = entry.get("p95_ms")
        ts_value = entry.get("ts") if isinstance(entry.get("ts"), (int, float)) else None
        ts_text = _format_timestamp(ts_value)
        source_text = _sanitize_text(entry.get("source"))
        stale_last = bool(entry.get("stale_last"))
        rate_count_val = entry.get("rate_limit_count")
        rate_count = int(rate_count_val) if isinstance(rate_count_val, (int, float)) else 0
        rate_avg = entry.get("rate_limit_avg_ms")
        rate_last = entry.get("rate_limit_last_ms")
        rate_reason = _format_rate_limit_reason(entry.get("rate_limit_last_reason"))

        icon = "🛟" if stale_last else "✅"
        if label.casefold() == "error":
            icon = "⚠️"

        parts = [f"{icon} {label}: {count_int} consultas"]
        if stale_count:
            parts.append(f"stale {stale_count}")
        if ok_count:
            if ok_ratio is not None:
                parts.append(f"OK {ok_count}/{count_int} ({ok_ratio:.0%})")
            else:
                parts.append(f"OK {ok_count}/{count_int}")
        if isinstance(avg_ms, (int, float)):
            parts.append(f"prom. {float(avg_ms):.0f} ms")
        if isinstance(last_ms, (int, float)):
            parts.append(f"último {float(last_ms):.0f} ms")
        if isinstance(p50_ms, (int, float)):
            parts.append(f"p50 {float(p50_ms):.0f} ms")
        if isinstance(p95_ms, (int, float)):
            parts.append(f"p95 {float(p95_ms):.0f} ms")
        if rate_count:
            rate_bits: list[str] = [f"limit {rate_count}"]
            if isinstance(rate_avg, (int, float)):
                rate_bits.append(f"prom. espera {float(rate_avg):.0f} ms")
            if isinstance(rate_last, (int, float)):
                rate_bits.append(f"últ. espera {float(rate_last):.0f} ms")
            if rate_reason:
                rate_bits.append(f"motivo {rate_reason}")
            parts.append(", ".join(rate_bits))
        if source_text:
            parts.append(f"fuente: {source_text}")
        parts.append(ts_text)

        lines.append(format_note(" • ".join(parts)))

    return lines


def _format_fx_api_status(data: Optional[dict]) -> str:
    if not data:
        return "_Sin llamadas a la API FX._"
    status = data.get("status")
    label = "API FX OK" if status == "success" else "API FX con errores"
    ts = _format_timestamp(_coerce_timestamp(data.get("last_fetch_ts") or data.get("ts")))
    elapsed = data.get("elapsed_ms")
    elapsed_txt = f"{float(elapsed):.0f} ms" if isinstance(elapsed, (int, float)) else "s/d"
    detail = data.get("error")
    detail_txt = f" — {detail}" if detail else ""
    prefix = "✅" if status == "success" else "⚠️"
    freshness_value = data.get("freshness")
    freshness_icon, freshness_label = _freshness_badge(freshness_value)
    parts = [f"{freshness_icon} {prefix} {label}"]
    parts.append(f"último fetch {ts}")
    parts.append(f"latencia {elapsed_txt}")
    if freshness_value is not None:
        parts.append(f"estado {freshness_label.lower()}")
    source_text = _sanitize_text(data.get("source"))
    if source_text:
        parts.append(f"origen {source_text}")
    return format_note(" • ".join(parts) + detail_txt)


def _format_fx_cache_status(data: Optional[dict]) -> str:
    if not data:
        return "_Sin uso de caché registrado._"
    mode = data.get("mode")
    icon = "✅" if mode == "hit" else "ℹ️"
    label = "Uso de caché" if mode == "hit" else "Actualización"
    ts = _format_timestamp(_coerce_timestamp(data.get("last_fetch_ts") or data.get("ts")))
    age = data.get("age")
    age_txt = f"{float(age):.0f}s" if isinstance(age, (int, float)) else "s/d"
    freshness_value = data.get("freshness")
    freshness_icon, freshness_label = _freshness_badge(freshness_value)
    detail_text = _sanitize_text(data.get("detail"))
    detail_suffix = f" — {detail_text}" if detail_text else ""
    parts = [f"{freshness_icon} {icon} {label}"]
    parts.append(f"último fetch {ts}")
    parts.append(f"edad {age_txt}")
    if freshness_value is not None:
        parts.append(f"estado {freshness_label.lower()}")
    hit_ratio = data.get("hit_ratio")
    if isinstance(hit_ratio, (int, float)):
        parts.append(f"hit ratio {float(hit_ratio):.0%}")
    return format_note(" • ".join(parts) + detail_suffix)


def _format_latency_line(label: str, data: Optional[dict]) -> str:
    if not data:
        return f"⚪️ - {label}: sin registro"
    freshness_value = data.get("freshness")
    freshness_icon, freshness_label = _freshness_badge(freshness_value)
    elapsed = data.get("elapsed_ms")
    elapsed_txt = f"{float(elapsed):.0f} ms" if isinstance(elapsed, (int, float)) else "s/d"
    parts = [f"{freshness_icon} - {label}: {elapsed_txt}"]
    source = data.get("source")
    if source:
        parts.append(f"fuente: {source}")
    count = data.get("count")
    if count is not None:
        parts.append(f"items: {count}")
    detail = data.get("detail")
    if detail:
        parts.append(detail)
    ts_value = _coerce_timestamp(data.get("last_fetch_ts") or data.get("ts"))
    parts.append(f"último fetch {_format_timestamp(ts_value)}")
    if freshness_value is not None:
        parts.append(f"estado {freshness_label.lower()}")
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


def _format_macro_history(
    attempts: Optional[Iterable[Mapping[str, Any]]],
    *,
    limit: int = 5,
) -> Iterable[str]:
    if not attempts:
        return []

    recent: list[Mapping[str, Any]] = []
    for entry in attempts:
        if isinstance(entry, Mapping):
            recent.append(entry)
    if not recent:
        return []

    lines: list[str] = []
    for attempt in reversed(recent[-limit:]):
        status = str(attempt.get("status_normalized") or attempt.get("status") or "unknown").casefold()
        label = str(attempt.get("provider_label") or attempt.get("label") or attempt.get("provider") or "desconocido")
        ts = _format_timestamp(attempt.get("ts"))
        elapsed = attempt.get("elapsed_ms")
        if isinstance(elapsed, (int, float)):
            elapsed_txt = f"{float(elapsed):.0f} ms"
        else:
            elapsed_txt = "s/d"
        detail = attempt.get("detail")
        detail_txt = f" — {detail}" if detail else ""
        missing_raw = attempt.get("missing_series")
        missing: list[str] = []
        if isinstance(missing_raw, str):
            candidate = missing_raw.strip()
            if candidate:
                missing = [candidate]
        elif isinstance(missing_raw, Iterable) and not isinstance(missing_raw, (bytes, bytearray, str)):
            missing = [str(item).strip() for item in missing_raw if str(item).strip()]
        missing_txt = f" • sin series: {', '.join(missing)}" if missing else ""
        fallback_flag = bool(attempt.get("fallback"))

        icon_map = {
            "success": "✅",
            "error": "⚠️",
            "disabled": "⛔️",
            "unavailable": "❌",
        }
        icon = icon_map.get(status, "ℹ️")
        if fallback_flag and status == "success":
            icon = "ℹ️"
        if status == "success" and not fallback_flag:
            status_label = "OK"
        elif fallback_flag:
            status_label = "Fallback"
        else:
            status_label = status.title()

        lines.append(format_note(f"{icon} {label}: {status_label} • {ts} ({elapsed_txt}){detail_txt}{missing_txt}"))

    return lines


def _format_macro_latency(buckets: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not isinstance(buckets, Mapping):
        return None

    counts = buckets.get("counts")
    ratios = buckets.get("ratios")
    if not isinstance(counts, Mapping):
        return None

    parts: list[str] = []
    for key, label in (("fast", "rápido"), ("medium", "medio"), ("slow", "lento")):
        count = counts.get(key)
        if not isinstance(count, (int, float)):
            continue
        count_int = int(count)
        if count_int <= 0:
            continue
        ratio_val = ratios.get(key) if isinstance(ratios, Mapping) else None
        if isinstance(ratio_val, (int, float)):
            parts.append(f"{label} {ratio_val:.0%} ({count_int})")
        else:
            parts.append(f"{label} ({count_int})")

    missing = counts.get("missing")
    if isinstance(missing, (int, float)) and int(missing) > 0:
        parts.append(f"sin dato {int(missing)}")

    if not parts:
        return None

    return "Latencia: " + " | ".join(parts)


def _format_macro_provider(summary: Mapping[str, Any]) -> str:
    label = str(summary.get("label") or summary.get("provider") or "desconocido")
    latest = summary.get("latest") if isinstance(summary.get("latest"), Mapping) else {}
    status = str(latest.get("status") or "desconocido")
    fallback_flag = bool(latest.get("fallback"))
    elapsed = latest.get("elapsed_ms")
    elapsed_txt = f"{float(elapsed):.0f} ms" if isinstance(elapsed, (int, float)) else "s/d"
    ts = _format_timestamp(latest.get("ts"))
    detail = latest.get("detail")
    detail_txt = f" — {detail}" if detail else ""

    if status == "error":
        icon = "⚠️"
        status_label = "Error reciente"
    elif fallback_flag:
        icon = "ℹ️"
        status_label = "Fallback"
    elif status == "success":
        icon = "✅"
        status_label = "OK"
    else:
        icon = "ℹ️"
        status_label = status

    total = summary.get("count")
    total_int = int(total) if isinstance(total, (int, float)) else 0
    status_counts = summary.get("status_counts") if isinstance(summary.get("status_counts"), Mapping) else {}
    status_ratios = summary.get("status_ratios") if isinstance(summary.get("status_ratios"), Mapping) else {}

    parts: list[str] = []
    success_count = status_counts.get("success")
    success_ratio = status_ratios.get("success")
    if isinstance(success_count, (int, float)):
        if isinstance(success_ratio, (int, float)) and total_int:
            parts.append(f"éxitos {success_ratio:.0%} ({int(success_count)}/{total_int})")
        else:
            parts.append(f"éxitos {int(success_count)}")

    for key, count_val in status_counts.items():
        if key in {"success", "error", "fallback"}:
            continue
        if not isinstance(count_val, (int, float)):
            continue
        ratio_val = status_ratios.get(key)
        label_txt = str(key)
        if isinstance(ratio_val, (int, float)) and total_int:
            parts.append(f"{label_txt} {ratio_val:.0%} ({int(count_val)}/{total_int})")
        else:
            parts.append(f"{label_txt} {int(count_val)}")

    error_count = summary.get("error_count")
    error_ratio = summary.get("error_ratio")
    if isinstance(error_count, (int, float)) and total_int:
        if isinstance(error_ratio, (int, float)):
            parts.append(f"errores {error_ratio:.0%} ({int(error_count)}/{total_int})")
        else:
            parts.append(f"errores {int(error_count)}/{total_int}")

    fallback_count = summary.get("fallback_count")
    fallback_ratio = summary.get("fallback_ratio")
    if isinstance(fallback_count, (int, float)) and total_int:
        if isinstance(fallback_ratio, (int, float)):
            parts.append(f"fallbacks {fallback_ratio:.0%} ({int(fallback_count)}/{total_int})")
        elif int(fallback_count):
            parts.append(f"fallbacks {int(fallback_count)}")

    latency_line = _format_macro_latency(summary.get("latency_buckets"))
    if latency_line:
        parts.append(latency_line)

    summary_txt = f" — {' • '.join(parts)}" if parts else ""
    return format_note(f"{icon} {label}: {status_label} • {ts} ({elapsed_txt}){detail_txt}{summary_txt}")


def _format_macro_overall(summary: Mapping[str, Any]) -> Optional[str]:
    total = summary.get("count")
    if not isinstance(total, (int, float)) or int(total) <= 0:
        return None
    total_int = int(total)
    status_counts = summary.get("status_counts") if isinstance(summary.get("status_counts"), Mapping) else {}
    status_ratios = summary.get("status_ratios") if isinstance(summary.get("status_ratios"), Mapping) else {}

    parts: list[str] = []
    success_count = status_counts.get("success")
    success_ratio = status_ratios.get("success")
    if isinstance(success_count, (int, float)) and isinstance(success_ratio, (int, float)):
        parts.append(f"éxitos {success_ratio:.0%} ({int(success_count)}/{total_int})")

    error_count = summary.get("error_count")
    error_ratio = summary.get("error_ratio")
    if isinstance(error_count, (int, float)) and isinstance(error_ratio, (int, float)):
        parts.append(f"errores {error_ratio:.0%} ({int(error_count)}/{total_int})")

    fallback_count = summary.get("fallback_count")
    fallback_ratio = summary.get("fallback_ratio")
    if isinstance(fallback_count, (int, float)) and isinstance(fallback_ratio, (int, float)):
        parts.append(f"fallbacks {fallback_ratio:.0%} ({int(fallback_count)}/{total_int})")

    latency_line = _format_macro_latency(summary.get("latency_buckets"))
    if latency_line:
        parts.append(latency_line)

    if not parts:
        return None

    return format_note(f"📊 Totales macro ({total_int}) — {' • '.join(parts)}")


def _format_macro_section(data: Optional[Mapping[str, Any]]) -> Iterable[str]:
    if not isinstance(data, Mapping):
        return ["_Sin datos macro registrados._"]

    providers = data.get("providers")
    if not isinstance(providers, Mapping) or not providers:
        return ["_Sin datos macro registrados._"]

    lines: list[str] = []
    overall = data.get("overall")
    if isinstance(overall, Mapping):
        overall_line = _format_macro_overall(overall)
        if overall_line:
            lines.append(overall_line)

    for key in sorted(providers):
        summary = providers.get(key)
        if not isinstance(summary, Mapping):
            continue
        lines.append(_format_macro_provider(summary))

    history_lines = list(_format_macro_history(data.get("attempts")))
    if history_lines:
        lines.append("_Historial de intentos recientes:_")
        lines.extend(history_lines)

    return lines or ["_Sin datos macro registrados._"]


def _sidebar_expander(label: str, *, expanded: bool = False):
    expander_fn = getattr(st.sidebar, "expander", None)
    if callable(expander_fn):
        return expander_fn(label, expanded=expanded)
    return st.expander(label, expanded=expanded)


def _container_expander(container: Any, label: str, *, expanded: bool = False):
    expander_fn = getattr(container, "expander", None)
    if callable(expander_fn):
        return expander_fn(label, expanded=expanded)
    return _sidebar_expander(label, expanded=expanded)


def _format_tab_latency_entry(key: str, stats: Mapping[str, Any]) -> str:
    label = _TAB_LABELS.get(key, str(stats.get("label") or key).title())
    parts: list[str] = []

    avg = stats.get("avg")
    if isinstance(avg, (int, float)):
        parts.append(f"μ {float(avg):.0f} ms")

    percentiles = stats.get("percentiles")
    if isinstance(percentiles, Mapping):
        for name, display in (
            ("p50", "P50"),
            ("p90", "P90"),
            ("p95", "P95"),
            ("p99", "P99"),
        ):
            value = percentiles.get(name)
            if isinstance(value, (int, float)):
                parts.append(f"{display} {float(value):.0f} ms")

    total = stats.get("total")
    total_int = int(total) if isinstance(total, (int, float)) else 0
    status_counts = stats.get("status_counts") if isinstance(stats.get("status_counts"), Mapping) else {}
    status_ratios = stats.get("status_ratios") if isinstance(stats.get("status_ratios"), Mapping) else {}

    success_count = status_counts.get("success")
    if isinstance(success_count, (int, float)) and total_int:
        success_ratio = status_ratios.get("success")
        if isinstance(success_ratio, (int, float)):
            parts.append(f"OK {int(success_count)}/{total_int} ({success_ratio:.0%})")
        else:
            parts.append(f"OK {int(success_count)}/{total_int}")

    error_ratio_value = stats.get("error_ratio") if isinstance(stats.get("error_ratio"), (int, float)) else None

    error_count = stats.get("error_count")
    if isinstance(error_count, (int, float)) and total_int:
        if isinstance(error_ratio_value, (int, float)):
            parts.append(f"errores {int(error_count)}/{total_int} ({error_ratio_value:.0%})")
        else:
            parts.append(f"errores {int(error_count)}/{total_int}")

    budget_value = stats.get("error_budget")
    if not isinstance(budget_value, (int, float)):
        budget_value = error_ratio_value
    if isinstance(budget_value, (int, float)):
        color = "green" if budget_value <= 0.05 else "red"
        parts.append(f":{color}[Budget {budget_value:.0%}]")

    missing_count = stats.get("missing_count")
    if isinstance(missing_count, (int, float)) and int(missing_count) > 0:
        parts.append(f"sin dato {int(missing_count)}")

    if not parts:
        parts.append("sin métricas registradas")

    return format_note(f"⏱️ {label} — {' • '.join(parts)}")


def _format_tab_latency_section(data: Optional[Mapping[str, Any]]) -> Iterable[str]:
    if not isinstance(data, Mapping) or not data:
        return ["_Sin latencias registradas._"]

    lines: list[str] = []
    for key in sorted(data):
        stats = data.get(key)
        if not isinstance(stats, Mapping):
            continue
        lines.append(_format_tab_latency_entry(key, stats))
    return lines or ["_Sin latencias registradas._"]


def _format_adapter_fallback_line(adapter_label: str, provider_stats: Mapping[str, Any]) -> str:
    provider_label = str(provider_stats.get("label") or "desconocido")
    count = provider_stats.get("count")
    fallback_count = provider_stats.get("fallback_count")
    fallback_ratio = provider_stats.get("fallback_ratio")
    status_counts = (
        provider_stats.get("status_counts") if isinstance(provider_stats.get("status_counts"), Mapping) else {}
    )
    status_ratios = (
        provider_stats.get("status_ratios") if isinstance(provider_stats.get("status_ratios"), Mapping) else {}
    )

    total_int = int(count) if isinstance(count, (int, float)) else 0
    metrics: list[str] = []
    if isinstance(fallback_count, (int, float)) and total_int:
        if isinstance(fallback_ratio, (int, float)):
            metrics.append(f"fallback {fallback_ratio:.0%} ({int(fallback_count)}/{total_int})")
        elif int(fallback_count):
            metrics.append(f"fallback {int(fallback_count)}/{total_int}")

    success_count = status_counts.get("success")
    if isinstance(success_count, (int, float)) and total_int:
        success_ratio = status_ratios.get("success")
        if isinstance(success_ratio, (int, float)):
            metrics.append(f"OK {int(success_count)}/{total_int} ({success_ratio:.0%})")
        else:
            metrics.append(f"OK {int(success_count)}/{total_int}")

    error_count = status_counts.get("error")
    if isinstance(error_count, (int, float)) and total_int:
        error_ratio = status_ratios.get("error")
        if isinstance(error_ratio, (int, float)):
            metrics.append(f"errores {int(error_count)}/{total_int} ({error_ratio:.0%})")
        else:
            metrics.append(f"errores {int(error_count)}/{total_int}")

    if not metrics:
        metrics.append("sin métricas registradas")

    return format_note(f"🛟 {adapter_label} → {provider_label} — {' • '.join(metrics)}")


def _format_adapter_fallback_section(
    data: Optional[Mapping[str, Any]],
) -> Iterable[str]:
    if not isinstance(data, Mapping):
        return ["_Sin registros de fallbacks._"]

    adapters = data.get("adapters")
    lines: list[str] = []
    if isinstance(adapters, Mapping):
        for key in sorted(adapters):
            adapter_entry = adapters.get(key)
            if not isinstance(adapter_entry, Mapping):
                continue
            label = str(adapter_entry.get("label") or key).strip() or str(key)
            providers = adapter_entry.get("providers")
            if not isinstance(providers, Mapping):
                continue
            for provider_key in sorted(providers):
                provider_stats = providers.get(provider_key)
                if not isinstance(provider_stats, Mapping):
                    continue
                lines.append(_format_adapter_fallback_line(label, provider_stats))

    if not lines:
        return ["_Sin registros de fallbacks._"]

    provider_totals = data.get("providers")
    if isinstance(provider_totals, Mapping):
        aggregated: list[str] = []
        for provider_key in sorted(provider_totals):
            stats = provider_totals.get(provider_key)
            if not isinstance(stats, Mapping):
                continue
            label = str(stats.get("label") or provider_key).strip() or str(provider_key)
            count = stats.get("count")
            total_int = int(count) if isinstance(count, (int, float)) else 0
            fallback_count = stats.get("fallback_count")
            fallback_ratio = stats.get("fallback_ratio")
            metrics: list[str] = []
            if isinstance(fallback_count, (int, float)) and total_int:
                if isinstance(fallback_ratio, (int, float)):
                    metrics.append(f"fallback {fallback_ratio:.0%} ({int(fallback_count)}/{total_int})")
                elif int(fallback_count):
                    metrics.append(f"fallback {int(fallback_count)}/{total_int}")
            status_counts = stats.get("status_counts") if isinstance(stats.get("status_counts"), Mapping) else {}
            status_ratios = stats.get("status_ratios") if isinstance(stats.get("status_ratios"), Mapping) else {}
            success_count = status_counts.get("success")
            if isinstance(success_count, (int, float)) and total_int:
                success_ratio = status_ratios.get("success")
                if isinstance(success_ratio, (int, float)):
                    metrics.append(f"OK {int(success_count)}/{total_int} ({success_ratio:.0%})")
                else:
                    metrics.append(f"OK {int(success_count)}/{total_int}")
            if metrics:
                aggregated.append(format_note(f"Σ {label} — {' • '.join(metrics)}"))
        if aggregated:
            lines.append("_Totales por proveedor:_")
            lines.extend(aggregated)

    return lines


def _render_health_panel(host: Any, metrics: Mapping[str, Any]) -> None:
    """Render the health summary panel inside the given container."""
    host.header(f"🩺 Healthcheck (versión {__version__})")
    host.caption("Monitorea la procedencia y el rendimiento de los datos cargados.")

    env_badge = _format_environment_badge(metrics.get("environment_snapshot"))

    auth_metrics = None
    for key in ("authentication", "auth_state", "auth"):
        candidate = metrics.get(key)
        if candidate is not None:
            auth_metrics = candidate
            break
    auth_lines = list(_format_authentication_section(auth_metrics))
    iol_summary = _format_iol_status(metrics.get("iol_refresh"))
    session_lines = list(_format_session_monitoring(metrics.get("session_monitoring")))

    overview_section = host.container(border=True)
    with overview_section:
        st.markdown("#### 🔍 Resumen operativo")
        if env_badge:
            st.markdown(env_badge)
        if auth_lines:
            st.markdown(auth_lines[0])
        else:
            st.markdown("_Sin métricas de autenticación._")
        st.markdown(iol_summary)
        if session_lines:
            st.markdown(session_lines[0])
        if (len(auth_lines) > 1) or (len(session_lines) > 1):
            with _container_expander(overview_section, "Detalle de autenticación y sesión", expanded=False):
                if auth_lines:
                    for line in auth_lines:
                        st.markdown(line)
                if session_lines:
                    if auth_lines:
                        st.markdown("---")
                    for line in session_lines:
                        st.markdown(line)

    _render_investor_profile_section(host)

    snapshot_metrics = None
    for key in ("snapshot", "snapshots", "snapshot_status", "portfolio_snapshot"):
        candidate = metrics.get(key)
        if candidate is not None:
            snapshot_metrics = candidate
            break

    yfinance_summary = _format_yfinance_status(metrics.get("yfinance"))
    snapshot_lines = list(_format_snapshot_section(snapshot_metrics))
    quote_lines = list(_format_quote_providers(metrics.get("quote_providers")))
    fx_lines = list(_format_fx_section(metrics.get("fx_api"), metrics.get("fx_cache")))
    macro_lines = list(_format_macro_section(metrics.get("macro_api")))

    data_section = host.container(border=True)
    with data_section:
        st.markdown("#### 📊 Salud de datos")
        st.markdown(yfinance_summary)
        if snapshot_lines:
            st.markdown(snapshot_lines[0])
        if quote_lines:
            st.markdown(quote_lines[0])
        if fx_lines:
            st.markdown(fx_lines[0])
        if macro_lines:
            st.markdown(macro_lines[0])

        has_data_details = any(len(lines) > 1 for lines in (snapshot_lines, quote_lines, fx_lines, macro_lines))
        if has_data_details:
            with _container_expander(data_section, "Detalle de proveedores y cachés"):
                st.markdown("**📈 Yahoo Finance**")
                st.markdown(yfinance_summary)
                if snapshot_lines:
                    st.markdown("**💾 Snapshots**")
                    for line in snapshot_lines:
                        st.markdown(line)
                if quote_lines:
                    st.markdown("**💹 Cotizaciones**")
                    for line in quote_lines:
                        st.markdown(line)
                if fx_lines:
                    st.markdown("**💱 FX**")
                    for line in fx_lines:
                        st.markdown(line)
                if macro_lines:
                    st.markdown("**🌐 Macro / Datos externos**")
                    for line in macro_lines:
                        st.markdown(line)

    risk_lines = list(_format_risk_section(metrics.get("risk_incidents")))
    risk_detail_lines = list(_format_risk_detail_section(metrics.get("risk_incidents")))
    tab_latency_lines = list(_format_tab_latency_section(metrics.get("tab_latencies")))
    fallback_lines = list(_format_adapter_fallback_section(metrics.get("adapter_fallbacks")))

    observability_section = host.container(border=True)
    with observability_section:
        st.markdown("#### 🚨 Riesgo y observabilidad")
        if risk_lines:
            st.markdown(risk_lines[0])
        if tab_latency_lines:
            st.markdown(tab_latency_lines[0])
        if fallback_lines:
            st.markdown(fallback_lines[0])

        has_risk_details = (len(risk_lines) > 1) or (
            risk_detail_lines and not (len(risk_detail_lines) == 1 and "_Sin" in risk_detail_lines[0])
        )
        if has_risk_details:
            with _container_expander(observability_section, "Detalle de incidencias de riesgo"):
                for line in risk_lines:
                    st.markdown(line)
                if risk_detail_lines:
                    st.markdown("---")
                    for line in risk_detail_lines:
                        st.markdown(line)
        if len(tab_latency_lines) > 1:
            with _container_expander(observability_section, "Latencias por pestaña"):
                for line in tab_latency_lines:
                    st.markdown(line)
        if len(fallback_lines) > 1:
            with _container_expander(observability_section, "Fallbacks por adaptador"):
                for line in fallback_lines:
                    st.markdown(line)

    diagnostics_section = host.container(border=True)
    with diagnostics_section:
        st.markdown("#### 🧪 Diagnóstico y soporte")
        _render_recent_stats(diagnostics_section, metrics)

        diagnostics_lines = list(_format_diagnostics_section(metrics.get("diagnostics")))
        if diagnostics_lines:
            with _container_expander(diagnostics_section, "Diagnóstico inicial"):
                for line in diagnostics_lines:
                    st.markdown(line)

        dependencies_lines = list(_format_dependencies_section(metrics.get("dependencies")))
        if dependencies_lines:
            with _container_expander(diagnostics_section, "Dependencias críticas"):
                for line in dependencies_lines:
                    st.markdown(line)

        latency_lines = list(_format_latency_section(metrics.get("portfolio"), metrics.get("quotes")))
        if latency_lines:
            with _container_expander(diagnostics_section, "Latencias de cálculo"):
                for line in latency_lines:
                    st.markdown(format_note(line))

        log_path = Path("analysis.log")
        log_bytes = None
        log_summary = ""
        try:
            log_path.stat()
        except FileNotFoundError:
            log_summary = "_No se encontró analysis.log._"
        except OSError:
            log_summary = "_No se pudo leer analysis.log._"
        else:
            try:
                log_bytes = log_path.read_bytes()
            except OSError:
                log_summary = "_No se pudo leer analysis.log._"
            else:
                log_summary = format_note("📦 analysis.log listo para descargar")

        if log_summary:
            st.markdown(log_summary)
        if log_bytes is not None:
            with _container_expander(diagnostics_section, "Descargar logs"):
                st.download_button(
                    "⬇️ Descargar analysis.log",
                    log_bytes,
                    file_name="analysis.log",
                )


def _resolve_health_metrics(metrics: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if isinstance(metrics, Mapping):
        return metrics
    return get_health_metrics()


def render_health_sidebar(*, metrics: Optional[Mapping[str, Any]] = None) -> None:
    """Render the health summary inside Streamlit's sidebar."""

    resolved = _resolve_health_metrics(metrics)
    _render_health_panel(st.sidebar, resolved)


def _log_missing_panel(label: str, module_path: str) -> None:
    key = f"{module_path}:{label}"
    if key in _MISSING_PANELS_LOGGED:
        return
    _MISSING_PANELS_LOGGED.add(key)
    logger.info("[monitoring] acceso sin render disponible label=%s module=%s", label, module_path)


def _find_render_attribute(module: ModuleType) -> str | None:
    exports = getattr(module, "__all__", None)
    candidates: list[str] = []
    if isinstance(exports, (list, tuple, set)):
        for name in exports:
            if isinstance(name, str) and name.startswith("render_"):
                candidates.append(name)
    if not candidates:
        candidates = [name for name in dir(module) if name.startswith("render_")]
    for name in candidates:
        attr = getattr(module, name, None)
        if callable(attr):
            return name
    return None


def _resolve_monitoring_renderer(module_path: str, label: str) -> tuple[str, str] | None:
    try:
        module = importlib.import_module(module_path)
    except Exception:
        _log_missing_panel(label, module_path)
        return None

    attribute = _find_render_attribute(module)
    if attribute is None:
        _log_missing_panel(label, module_path)
        return None
    return module.__name__, attribute


def _activate_monitoring_panel(module_path: str, attribute: str, label: str) -> None:
    state = getattr(st, "session_state", None)
    if state is None:
        return
    state[_MONITORING_PANEL_STATE_KEY] = {
        "module": module_path,
        "attr": attribute,
        "label": label,
    }
    logger.info(
        "[monitoring] panel activado label=%s module=%s attr=%s",
        label,
        module_path,
        attribute,
    )


def _clear_active_monitoring_panel() -> None:
    state = getattr(st, "session_state", None)
    if state is None:
        return
    payload = state.pop(_MONITORING_PANEL_STATE_KEY, None)
    if isinstance(payload, Mapping):
        module_path = payload.get("module")
        label = payload.get("label") or module_path or ""
        logger.info(
            "[monitoring] panel cerrado label=%s module=%s",
            label,
            module_path,
        )


def _get_active_monitoring_panel() -> dict[str, str] | None:
    state = getattr(st, "session_state", None)
    if state is None:
        return None
    payload = state.get(_MONITORING_PANEL_STATE_KEY)
    if isinstance(payload, Mapping):
        module_path = payload.get("module")
        attribute = payload.get("attr")
        if isinstance(module_path, str) and isinstance(attribute, str):
            label = str(payload.get("label") or module_path)
            return {
                "module": module_path,
                "attr": attribute,
                "label": label,
            }
    return None


def _import_renderer_callable(module_path: str, attribute: str, label: str) -> Callable[[], None] | None:
    try:
        module = importlib.import_module(module_path)
    except Exception:
        _log_missing_panel(label, module_path)
        return None

    candidate = getattr(module, attribute, None)
    if callable(candidate):
        module_streamlit = getattr(module, "st", None)
        if module_streamlit is not st:
            try:
                setattr(module, "st", st)
            except Exception:
                logger.debug(
                    "[monitoring] no se pudo sincronizar streamlit en módulo label=%s module=%s",
                    label,
                    module_path,
                )
        return candidate

    _log_missing_panel(label, module_path)
    return None


def _render_active_monitoring_panel(selection: Mapping[str, str]) -> bool:
    module_path = selection.get("module")
    attribute = selection.get("attr")
    label = selection.get("label") or module_path or ""
    if not isinstance(module_path, str) or not isinstance(attribute, str):
        _clear_active_monitoring_panel()
        return False

    renderer = _import_renderer_callable(module_path, attribute, str(label))
    if renderer is None:
        st.warning("El panel seleccionado no está disponible actualmente.")
        _clear_active_monitoring_panel()
        return False

    if st.button("⬅️ Volver al monitoreo", key="monitoring_back_button"):
        _clear_active_monitoring_panel()
        return False

    renderer()

    state = getattr(st, "session_state", None)
    stop_callable = getattr(st, "stop", None)
    if callable(stop_callable):
        try:
            stop_callable()
        except RuntimeError as exc:
            message = str(exc)
            if "streamlit.stop" not in message.lower():
                raise
        else:
            if state is not None:
                state[_MONITORING_RENDERED_FLAG] = True
            return True
    if state is not None:
        state[_MONITORING_RENDERED_FLAG] = True
    return True


def _should_abort_after_monitoring_panel() -> bool:
    state = getattr(st, "session_state", None)
    if state is None:
        return False
    try:
        flag = state.pop(_MONITORING_RENDERED_FLAG)
    except Exception:
        return False
    return bool(flag)


def _render_monitoring_shortcuts() -> bool:
    had_selection = _get_active_monitoring_panel() is not None
    for label, module_path in _MONITORING_SHORTCUTS:
        resolved = _resolve_monitoring_renderer(module_path, label)
        if resolved is None:
            safe_page_link(module_path, label=label)
            continue

        module_name, attribute = resolved

        def _activate(
            module_name: str = module_name,
            attribute: str = attribute,
            link_label: str = label,
        ) -> None:
            _activate_monitoring_panel(module_name, attribute, link_label)

        safe_page_link(
            module_path,
            label=label,
            render_fallback=_activate,
            prefer_inline=True,
        )

    return (not had_selection) and _get_active_monitoring_panel() is not None


def render_health_monitor_tab(container: Any, *, metrics: Optional[Mapping[str, Any]] = None) -> None:
    """Render the health summary within the provided tab container."""

    selection = _get_active_monitoring_panel()
    if selection is not None:
        rendered = _render_active_monitoring_panel(selection)
        if _should_abort_after_monitoring_panel():
            return
        if rendered:
            return
        selection = _get_active_monitoring_panel()

    shortcuts_container = container.container(border=True)
    with shortcuts_container:
        st.markdown("### 🔗 Recursos de monitoreo")
        st.caption("Accedé a paneles complementarios desde esta vista.")
        activated = _render_monitoring_shortcuts()
        if activated:
            selection = _get_active_monitoring_panel()
            if selection is not None:
                rendered = _render_active_monitoring_panel(selection)
                if _should_abort_after_monitoring_panel():
                    return
                if rendered:
                    return

    resolved = _resolve_health_metrics(metrics)
    control_hub = container.container(border=True)
    with control_hub:
        st.markdown("### 🎛️ Centro de control")
        st.caption("Gestioná filtros, sesión y apariencia desde una sola pestaña.")
    render_action_menu(container=control_hub)
    render_ui_controls(container=control_hub)

    symbols, types = get_controls_reference_data()
    render_controls_panel(symbols, types, container=control_hub)

    if hasattr(container, "divider"):
        container.divider()
    _render_health_panel(container, resolved)


def summarize_health_status(
    *, metrics: Optional[Mapping[str, Any]] = None
) -> tuple[str, str, Optional[str], str, Optional[float]]:
    """Return icon, label, detail, severity and last failure timestamp for health."""

    resolved = _resolve_health_metrics(metrics)

    diagnostics = resolved.get("diagnostics")
    status_value: Any = None
    detail_value: Any = None
    ts_value: Optional[float] = None
    failure_candidates: list[float] = []

    def _track_failure(entry: Mapping[str, Any] | None) -> None:
        if not isinstance(entry, Mapping):
            return

        status_key = _normalize_status_key(entry.get("status"))
        if status_key in _STATUS_SEVERITY_DANGER:
            ts_candidate = (
                _coerce_timestamp(entry.get("ts"))
                or _coerce_timestamp(entry.get("last_fetch_ts"))
                or _coerce_timestamp(entry.get("last_error_ts"))
                or _coerce_timestamp(entry.get("last_failure_ts"))
            )
            if ts_candidate is not None:
                failure_candidates.append(ts_candidate)

        for key in ("last_error_ts", "last_failure_ts", "last_failure", "last_error"):
            ts_candidate = _coerce_timestamp(entry.get(key))
            if ts_candidate is not None:
                failure_candidates.append(ts_candidate)

    if isinstance(diagnostics, Mapping):
        primary = diagnostics.get("initial")
        if isinstance(primary, Mapping):
            status_value = primary.get("status") or status_value
            detail_value = primary.get("detail") or primary.get("message") or detail_value
            ts_value = _coerce_timestamp(primary.get("ts")) or ts_value
            _track_failure(primary)

            checks = primary.get("checks")
            if isinstance(checks, Sequence):
                for check in checks:
                    if isinstance(check, Mapping):
                        _track_failure(check)
        status_value = status_value or diagnostics.get("status")
        detail_value = detail_value or diagnostics.get("detail") or diagnostics.get("message")
        ts_value = ts_value or _coerce_timestamp(diagnostics.get("ts"))
        _track_failure(diagnostics)

    if status_value is None:
        environment = resolved.get("environment_snapshot")
        if isinstance(environment, Mapping):
            status_value = environment.get("status")
            detail_value = detail_value or environment.get("detail") or environment.get("message")
            ts_value = ts_value or _coerce_timestamp(environment.get("ts"))
            _track_failure(environment)

    for key in ("authentication", "iol_refresh"):
        entry = resolved.get(key)
        if isinstance(entry, Mapping):
            _track_failure(entry)

    dependencies = resolved.get("dependencies")
    if isinstance(dependencies, Mapping):
        items = dependencies.get("items")
        if isinstance(items, Mapping):
            iterable = items.values()
        else:
            iterable = dependencies.values()
        for entry in iterable:
            if isinstance(entry, Mapping):
                _track_failure(entry)

    icon, label = _status_badge(status_value or "", default_label="Sin datos")
    severity = _categorize_status(status_value or "")

    detail_text = _sanitize_text(detail_value)
    if ts_value is not None:
        ts_text = _format_timestamp(ts_value)
        if ts_text:
            detail_text = f"{detail_text} • {ts_text}" if detail_text else ts_text

    last_failure_ts = max(failure_candidates) if failure_candidates else None

    return icon, label, detail_text, severity, last_failure_ts


__all__ = [
    "render_health_monitor_tab",
    "render_health_sidebar",
    "summarize_health_status",
]
