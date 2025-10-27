"""Centralized control for Streamlit rerun requests with debounce support."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from typing import Any

try:  # pragma: no cover - optional dependency in tests
    import streamlit as st
except Exception:  # pragma: no cover - Streamlit not available during some tests
    st = None  # type: ignore

from shared.telemetry import log_metric

from .rerun_trace import mark_event, safe_rerun
from .ui_flow import current_flow_id

logger = logging.getLogger(__name__)

_DEFAULT_DEBOUNCE_MS = 400.0
_STATE_KEY = "_pending_rerun_reason"
_TIMESTAMP_KEY = "_pending_rerun_timestamp"
_FLOW_KEY = "_pending_rerun_flow_id"
_FALLBACK_STATE: dict[str, Any] = {}

_REASON_PRIORITY: Mapping[str, int] = {
    "logout_requested": 0,
    "auth_logout_force_login": 0,
    "login_success": 5,
    "login_missing_credentials": 5,
    "login_invalid_credentials": 5,
    "login_network_error": 5,
    "login_app_error": 5,
    "login_runtime_error": 5,
    "login_unexpected_error": 5,
    "hydration_unlock": 10,
    "portfolio.extended_metrics_ready": 20,
    "lazy_fragment_ready": 30,
    "portfolio_autorefresh": 40,
}


def _session_state() -> dict[str, Any]:
    if st is None:
        return _FALLBACK_STATE
    try:
        return st.session_state
    except Exception:  # pragma: no cover - defensive safeguard
        return _FALLBACK_STATE


def _priority(reason: str) -> int:
    if not reason:
        return 100
    reason_key = reason.strip()
    if reason_key in _REASON_PRIORITY:
        return _REASON_PRIORITY[reason_key]
    for prefix, value in (
        ("login_", 5),
        ("logout", 0),
        ("hydration", 10),
        ("lazy_", 30),
        ("portfolio.", 20),
    ):
        if reason_key.startswith(prefix):
            return value
    return 100


def _ensure_queue(state: dict[str, Any]) -> list[str]:
    raw = state.get(_STATE_KEY)
    if isinstance(raw, list):
        queue = [str(item).strip() for item in raw if str(item or "").strip()]
    elif isinstance(raw, tuple | set):
        queue = [str(item).strip() for item in raw if str(item or "").strip()]
    else:
        queue = []
    state[_STATE_KEY] = queue
    return queue


def _select_reason(reasons: Iterable[str]) -> str:
    sanitized = [str(item).strip() for item in reasons if str(item or "").strip()]
    if not sanitized:
        return "generic_rerun"
    sorted_reasons = sorted(sanitized, key=_priority)
    return sorted_reasons[0]


def _log_debounce(
    *,
    delay_s: float,
    reason: str,
    flow_id: str | None,
    reasons: Iterable[str],
) -> None:
    try:
        duration_ms = max(float(delay_s), 0.0) * 1000.0
    except Exception:
        duration_ms = 0.0
    context = {
        "flow_id": flow_id or "unknown",
        "reason": reason,
        "pending_reasons": list(reasons),
    }
    try:
        log_metric(
            "ui.rerun_debounced_ms",
            duration_ms=duration_ms,
            context=context,
        )
    except Exception:  # pragma: no cover - telemetry best effort
        logger.debug("Unable to record ui.rerun_debounced_ms", exc_info=True)
    mark_event(
        "rerun_debounced",
        reason,
        {"delay_ms": duration_ms, "flow_id": flow_id, "pending": list(reasons)},
    )


def _log_coalesced(*, flow_id: str | None, reasons: Iterable[str]) -> None:
    reason_list = [str(item) for item in reasons if str(item)]
    if len(reason_list) <= 1:
        return
    context = {
        "flow_id": flow_id or "unknown",
        "pending_reasons": reason_list,
    }
    try:
        log_metric(
            "ui.rerun_coalesced_count",
            duration_ms=float(len(reason_list)),
            context=context,
        )
    except Exception:  # pragma: no cover - telemetry best effort
        logger.debug("Unable to record ui.rerun_coalesced_count", exc_info=True)
    mark_event(
        "rerun_coalesced",
        reason_list[0],
        {"count": len(reason_list), "flow_id": flow_id, "pending": reason_list},
    )


def request_rerun(reason: str, *, debounce_ms: float | None = None) -> None:
    """Request a Streamlit rerun applying debounce and coalescing logic."""

    reason_key = str(reason or "").strip()
    if not reason_key:
        logger.debug("Ignored rerun request without reason")
        return

    state = _session_state()
    queue = _ensure_queue(state)

    now = time.monotonic()
    debounce_window = max(float(debounce_ms or _DEFAULT_DEBOUNCE_MS), 0.0) / 1000.0
    last_ts = state.get(_TIMESTAMP_KEY)
    try:
        last_requested = float(last_ts)
    except (TypeError, ValueError):
        last_requested = 0.0
    flow_id = current_flow_id()

    within_window = last_requested > 0.0 and (now - last_requested) < debounce_window

    if not within_window:
        queue = [reason_key]
        state[_STATE_KEY] = queue
        state[_TIMESTAMP_KEY] = now
        state[_FLOW_KEY] = flow_id
        selected_reason = _select_reason(queue)
        logger.debug(
            "Dispatching rerun immediately reason=%s flow_id=%s", selected_reason, flow_id
        )
        safe_rerun(selected_reason)
        return

    if reason_key not in queue:
        queue.append(reason_key)
        queue.sort(key=_priority)
        state[_STATE_KEY] = queue
        _log_coalesced(flow_id=flow_id, reasons=queue)
    else:
        logger.debug("Duplicate rerun reason suppressed reason=%s flow_id=%s", reason_key, flow_id)

    state[_FLOW_KEY] = flow_id
    _log_debounce(
        delay_s=now - last_requested,
        reason=reason_key,
        flow_id=flow_id,
        reasons=queue,
    )


__all__ = ["request_rerun"]
