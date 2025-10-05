
from __future__ import annotations

"""Utilities to collect startup diagnostics and publish them to telemetry logs."""

from typing import Any, Mapping
import logging

import streamlit as st

from shared.time_provider import TimeProvider
from services.health import get_health_metrics


analysis_logger = logging.getLogger("analysis")

_STATUS_ICONS: Mapping[str, str] = {
    "success": "✅",
    "ok": "✅",
    "warning": "⚠️",
    "degraded": "⚠️",
    "error": "❌",
    "critical": "❌",
    "stale": "🟡",
    "info": "ℹ️",
}


def _normalize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(val) for key, val in value.items()}
    return str(value)


def _snapshot_session_state(state: Mapping[str, Any]) -> dict[str, Any]:
    session_id = state.get("session_id")
    snapshot: dict[str, Any] = {
        "id": str(session_id) if session_id is not None else None,
    }

    values: dict[str, Any] = {}
    flags: list[str] = []

    for key in sorted(state):
        if key == "session_id" or key.startswith("_"):
            continue
        value = state[key]
        if isinstance(value, bool):
            values[key] = value
            if value:
                flags.append(key)
            continue
        values[key] = _normalize_value(value)

    if values:
        snapshot["values"] = values
    if flags:
        snapshot["flags"] = sorted(flags)

    return snapshot


def _coerce_icon(entry: Mapping[str, Any], *, fallback: str) -> str:
    icon = entry.get("icon")
    if isinstance(icon, str) and icon.strip():
        return icon.strip()

    status = entry.get("status")
    status_key = str(status or "").strip().casefold()
    mapped = _STATUS_ICONS.get(status_key)
    if mapped:
        return mapped
    return "ℹ️"


def _resolve_label(entry: Mapping[str, Any], *, fallback: str) -> str:
    label = entry.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    name = entry.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return fallback


def _resolve_value(entry: Mapping[str, Any]) -> str:
    status = entry.get("status")
    detail = entry.get("detail")
    value = None
    if isinstance(status, str) and status.strip():
        value = status.strip()
    raw_value = entry.get("value")
    if value is None and isinstance(raw_value, str) and raw_value.strip():
        value = raw_value.strip()
    if value is None:
        numeric = entry.get("elapsed_ms") or entry.get("latency_ms")
        if isinstance(numeric, (int, float)):
            value = f"{float(numeric):.0f} ms"
    if value is None:
        value = "s/d"

    if isinstance(detail, str) and detail.strip():
        return f"{value} — {detail.strip()}"
    return value


def _collect_highlights(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    highlights: list[dict[str, Any]] = []
    if not isinstance(metrics, Mapping):
        return highlights

    for key in sorted(metrics):
        entry = metrics.get(key)
        if not isinstance(entry, Mapping):
            continue
        label = _resolve_label(entry, fallback=key.replace("_", " ").title())
        icon = _coerce_icon(entry, fallback=label)
        value = _resolve_value(entry)
        highlights.append({
            "id": key,
            "icon": icon,
            "label": label,
            "value": value,
        })

    return highlights


def run_startup_diagnostics() -> dict[str, Any]:
    """Gather startup diagnostics combining health metrics and session data."""

    metrics = get_health_metrics()
    session_snapshot = _snapshot_session_state(st.session_state)
    timestamp = TimeProvider.now()

    payload: dict[str, Any] = {
        "event": "startup.diagnostics",
        "timestamp": timestamp,
        "session": session_snapshot,
        "metrics": metrics,
        "highlights": _collect_highlights(metrics),
    }

    analysis_logger.info("startup.diagnostics", extra={"analysis": payload})
    return payload


__all__ = ["run_startup_diagnostics"]
