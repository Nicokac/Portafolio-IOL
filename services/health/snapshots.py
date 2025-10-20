"""Snapshot normalization and persistence helpers."""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional

from .constants import _ENVIRONMENT_SNAPSHOT_KEY, _SNAPSHOT_EVENT_KEY
from .session_adapter import get_store
from .telemetry import log_analysis_event
from .utils import (
    _clean_detail,
    _normalize_backend_details,
    _normalize_environment_snapshot,
)


def record_snapshot_event(
    *,
    kind: str,
    status: str,
    action: Optional[str] = None,
    storage_id: Optional[str] = None,
    detail: Optional[str] = None,
    backend: Optional[Mapping[str, Any]] = None,
) -> None:
    """Persist information about the latest snapshot interaction."""

    kind_text = str(kind or "generic").strip() or "generic"
    status_text = str(status or "unknown").strip() or "unknown"
    action_text = str(action or "").strip()
    storage_id_text = str(storage_id or "").strip()
    detail_text = _clean_detail(detail)

    event: Dict[str, Any] = {"kind": kind_text, "status": status_text, "ts": time.time()}
    if action_text:
        event["action"] = action_text
    if storage_id_text:
        event["storage_id"] = storage_id_text
    if detail_text:
        event["detail"] = detail_text

    backend_details = _normalize_backend_details(backend or {})
    if backend_details:
        event["backend"] = backend_details

    store = get_store()
    store[_SNAPSHOT_EVENT_KEY] = event


def record_environment_snapshot(snapshot: Mapping[str, Any]) -> None:
    """Persist the latest environment snapshot for diagnostics."""

    if not isinstance(snapshot, Mapping):
        return

    normalized_snapshot = _normalize_environment_snapshot(snapshot)
    entry: Dict[str, Any] = {"ts": time.time(), "snapshot": normalized_snapshot}

    store = get_store()
    store[_ENVIRONMENT_SNAPSHOT_KEY] = entry

    metrics: Dict[str, Any] = {}
    cpu_info = normalized_snapshot.get("cpu")
    if isinstance(cpu_info, Mapping):
        logical = cpu_info.get("logical_count")
        if isinstance(logical, (int, float)):
            metrics["cpu_logical"] = logical
    memory_info = normalized_snapshot.get("memory")
    if isinstance(memory_info, Mapping):
        total_mb = memory_info.get("total_mb")
        if isinstance(total_mb, (int, float)):
            metrics["memory_total_mb"] = total_mb

    if metrics:
        log_analysis_event("environment_snapshot", entry, metrics)


def snapshot_event_summary(raw_event: Any) -> Dict[str, Any]:
    """Normalize the stored snapshot event for downstream consumers."""

    if not isinstance(raw_event, Mapping):
        return {}

    summary: Dict[str, Any] = {}
    kind = raw_event.get("kind")
    if isinstance(kind, str) and kind.strip():
        summary["kind"] = kind.strip()
    status = raw_event.get("status")
    if isinstance(status, str) and status.strip():
        summary["status"] = status.strip()
    action = raw_event.get("action")
    if isinstance(action, str) and action.strip():
        summary["action"] = action.strip()
    storage = raw_event.get("storage_id")
    if isinstance(storage, str) and storage.strip():
        summary["storage_id"] = storage.strip()
    detail = _clean_detail(raw_event.get("detail"))
    if detail:
        summary["detail"] = detail

    ts_value = raw_event.get("ts")
    try:
        summary["ts"] = float(ts_value)
    except (TypeError, ValueError):
        pass

    backend = _normalize_backend_details(raw_event.get("backend"))
    if backend:
        summary["backend"] = backend

    return summary


__all__ = [
    "record_environment_snapshot",
    "record_snapshot_event",
    "snapshot_event_summary",
]
