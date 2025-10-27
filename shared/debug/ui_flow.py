"""Helpers to correlate Streamlit UI cycles with background activity."""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from typing import Iterator

try:  # pragma: no cover - optional dependency in tests
    import streamlit as st
except Exception:  # pragma: no cover - Streamlit not available during some tests
    st = None  # type: ignore

from .rerun_trace import mark_event

_DEBUG_MONITORING_FREEZE = os.getenv("DEBUG_MONITORING_FREEZE", "0") == "1"


def current_flow_id(default: str | None = None) -> str | None:
    """Return the current UI flow identifier stored in session state."""

    if st is None:
        return default
    try:
        flow_id = st.session_state.get("_ui_flow_id")
    except Exception:
        return default
    if isinstance(flow_id, str) and flow_id:
        return flow_id
    return default


def ensure_flow_id() -> str:
    """Ensure the session state holds a flow identifier and return it."""

    existing = current_flow_id()
    if existing:
        return existing
    return start_ui_flow("auto")


def start_ui_flow(reason: str, *, force_new: bool = False) -> str:
    """Assign a new flow identifier to the current session."""

    if st is None:
        flow_id = f"no-session-{uuid.uuid4().hex[:8]}"
        mark_event("ui_flow_start", reason, {"flow_id": flow_id, "force": force_new})
        return flow_id
    if not force_new:
        try:
            existing = st.session_state.get("_ui_flow_id")
        except Exception:
            existing = None
        if isinstance(existing, str) and existing:
            mark_event("ui_flow_resume", reason, {"flow_id": existing})
            return existing
    flow_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    try:
        st.session_state["_ui_flow_id"] = flow_id
        st.session_state["_ui_flow_started_at"] = time.time()
    except Exception:
        pass
    mark_event("ui_flow_start", reason, {"flow_id": flow_id, "force": force_new})
    return flow_id


@contextmanager
def background_job(job_name: str, *, detail: str | None = None) -> Iterator[dict[str, str | float | None]]:
    """Context manager to log background job execution tied to the current flow."""

    flow_id = current_flow_id()
    start = time.time()
    mark_event(
        "background_job_start",
        job_name,
        {
            "flow_id": flow_id,
            "detail": detail,
        },
    )
    try:
        yield {"flow_id": flow_id, "job": job_name, "detail": detail}
    except Exception as exc:
        mark_event(
            "background_job_error",
            job_name,
            {"flow_id": flow_id, "detail": detail, "error": repr(exc)},
        )
        raise
    else:
        duration = time.time() - start
        mark_event(
            "background_job_complete",
            job_name,
            {"flow_id": flow_id, "detail": detail, "duration": duration},
        )


def freeze_heavy_tasks() -> bool:
    """Return ``True`` when heavy tasks should be paused during monitoring."""

    return _DEBUG_MONITORING_FREEZE


__all__ = [
    "background_job",
    "current_flow_id",
    "ensure_flow_id",
    "freeze_heavy_tasks",
    "start_ui_flow",
]
