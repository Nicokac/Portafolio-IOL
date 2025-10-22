"""Streamlit session helpers for health metrics."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Mapping, Optional

import streamlit as st

from .constants import (
    _ACTIVE_SESSIONS_KEY,
    _HEALTH_KEY,
    _LAST_HTTP_ERROR_KEY,
    _LOGIN_TO_RENDER_STATS_KEY,
    _SESSION_MONITORING_KEY,
)
from .telemetry import log_analysis_event
from .utils import (
    _as_optional_float,
    _as_optional_int,
    _clean_detail,
    _normalize_metadata,
)

logger = logging.getLogger(__name__)


def get_store() -> Dict[str, Any]:
    """Return the mutable health metrics store from the session state."""
    return st.session_state.setdefault(_HEALTH_KEY, {})


def ensure_session_monitoring_store(store: Dict[str, Any]) -> Dict[str, Any]:
    raw_monitoring = store.get(_SESSION_MONITORING_KEY)
    if isinstance(raw_monitoring, Mapping):
        monitoring = dict(raw_monitoring)
    else:
        monitoring = {}
    if monitoring is not raw_monitoring:
        store[_SESSION_MONITORING_KEY] = monitoring
    return monitoring


def record_session_started(
    session_id: str,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Track the start of an interactive session for health dashboards."""

    session_key = str(session_id or "unknown").strip() or "unknown"
    now = time.time()
    store = get_store()
    monitoring = ensure_session_monitoring_store(store)

    active_raw = monitoring.get(_ACTIVE_SESSIONS_KEY)
    if isinstance(active_raw, Mapping):
        active_sessions = dict(active_raw)
    else:
        active_sessions = {}

    entry: Dict[str, Any] = {"session_id": session_key, "ts": now}
    metadata_dict = _normalize_metadata(metadata)
    if metadata_dict:
        entry["metadata"] = metadata_dict

    active_sessions[session_key] = entry
    monitoring[_ACTIVE_SESSIONS_KEY] = active_sessions
    monitoring["total_session_starts"] = int(monitoring.get("total_session_starts", 0) or 0) + 1
    monitoring["active_sessions_ts"] = now

    log_analysis_event(
        "session_started",
        entry,
        {
            "active_sessions": len(active_sessions),
            "total_session_starts": monitoring["total_session_starts"],
        },
    )


def record_login_to_render(
    elapsed_seconds: float,
    *,
    session_id: Optional[str] = None,
) -> None:
    """Record timing from login to first render for UX monitoring."""

    elapsed_value = _as_optional_float(elapsed_seconds)
    if elapsed_value is None or elapsed_value < 0:
        return

    now = time.time()
    store = get_store()
    monitoring = ensure_session_monitoring_store(store)
    stats_raw = monitoring.get(_LOGIN_TO_RENDER_STATS_KEY)
    if isinstance(stats_raw, Mapping):
        stats = dict(stats_raw)
    else:
        stats = {"count": 0, "sum": 0.0, "sum_sq": 0.0}

    stats["count"] = int(stats.get("count", 0) or 0) + 1
    stats["sum"] = float(stats.get("sum", 0.0) or 0.0) + elapsed_value
    stats["sum_sq"] = float(stats.get("sum_sq", 0.0) or 0.0) + elapsed_value * elapsed_value
    stats["last_value"] = elapsed_value
    stats["last_ts"] = now
    if session_id is not None:
        stats["last_session_id"] = str(session_id)

    monitoring[_LOGIN_TO_RENDER_STATS_KEY] = stats

    avg = stats["sum"] / stats["count"] if stats["count"] else None
    latest: Dict[str, Any] = {"value": stats["last_value"], "ts": now}
    session_ref = stats.get("last_session_id")
    if isinstance(session_ref, str) and session_ref.strip():
        latest["session_id"] = session_ref.strip()
    metrics: Dict[str, Any] = {"count": stats["count"]}
    if avg is not None:
        metrics["avg"] = avg
    log_analysis_event(
        "login_to_render",
        latest,
        metrics,
    )


def record_http_error(
    status_code: int,
    *,
    method: Optional[str] = None,
    url: Optional[str] = None,
    detail: Optional[str] = None,
) -> None:
    """Track the latest HTTP error encountered by the UI."""

    code_value = _as_optional_int(status_code)
    if code_value is None:
        return

    now = time.time()
    store = get_store()
    monitoring = ensure_session_monitoring_store(store)

    entry: Dict[str, Any] = {"status_code": code_value, "ts": now}
    method_text = str(method or "").strip()
    if method_text:
        entry["method"] = method_text
    url_text = str(url or "").strip()
    if url_text:
        entry["url"] = url_text
    detail_text = _clean_detail(detail)
    if detail_text:
        entry["detail"] = detail_text

    monitoring[_LAST_HTTP_ERROR_KEY] = entry
    monitoring["http_error_count"] = int(monitoring.get("http_error_count", 0) or 0) + 1

    log_analysis_event(
        "http_error",
        entry,
        {"http_error_count": monitoring["http_error_count"]},
    )


__all__ = [
    "ensure_session_monitoring_store",
    "get_store",
    "record_http_error",
    "record_login_to_render",
    "record_session_started",
    "st",
]
