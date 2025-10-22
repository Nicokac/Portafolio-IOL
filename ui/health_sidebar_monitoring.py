from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import streamlit as st

"""Render helpers for the monitoring sidebar that surfaces startup diagnostics."""


DEFAULT_DOWNLOAD_LABEL = "Descargar diagnóstico"
DEFAULT_FILE_NAME = "startup_diagnostics.json"


def _format_highlight(entry: Mapping[str, Any]) -> str:
    icon = str(entry.get("icon") or "•").strip() or "•"
    label = str(entry.get("label") or entry.get("id") or "").strip()
    value = entry.get("value")
    if isinstance(value, (int, float)):
        value_text = f"{value}"
    elif value is None:
        value_text = "s/d"
    else:
        value_text = str(value)
    if label:
        return f"{icon} **{label}:** {value_text}"
    return f"{icon} {value_text}"


def _render_session_section(session: Mapping[str, Any]) -> None:
    values = session.get("values")
    if isinstance(values, Mapping) and values:
        st.sidebar.markdown("#### Sesión")
        for key in sorted(values):
            st.sidebar.markdown(f"• {key}: {values[key]}")

    flags = session.get("flags")
    if isinstance(flags, Sequence):
        active = [str(flag) for flag in flags if str(flag).strip()]
    else:
        active = []
    if active:
        st.sidebar.markdown("#### Flags activos")
        for flag in active:
            st.sidebar.markdown(f"✅ {flag}")


def render_monitoring_sidebar(payload: Mapping[str, Any]) -> None:
    """Render the monitoring sidebar based on the diagnostics payload."""

    st.sidebar.header("🩺 Diagnóstico de arranque")
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, str) and timestamp.strip():
        st.sidebar.caption(f"Generado el {timestamp.strip()}")

    highlights = payload.get("highlights")
    if isinstance(highlights, Sequence):
        for entry in highlights:
            if isinstance(entry, Mapping):
                st.sidebar.markdown(_format_highlight(entry))

    session = payload.get("session")
    if isinstance(session, Mapping):
        _render_session_section(session)

    export_label = payload.get("download_label")
    label = str(export_label).strip() if isinstance(export_label, str) else DEFAULT_DOWNLOAD_LABEL
    file_name = payload.get("download_file_name")
    if not isinstance(file_name, str) or not file_name.strip():
        file_name = DEFAULT_FILE_NAME

    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    st.download_button(
        label,
        data=data,
        file_name=file_name,
        mime="application/json",
        key="startup_diagnostics",
    )


__all__ = ["render_monitoring_sidebar"]
