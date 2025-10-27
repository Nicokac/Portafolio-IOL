"""Utilities to trace Streamlit reruns and stop events for diagnostics."""

from __future__ import annotations

import os
import threading
import time
import traceback
from collections import deque
from typing import Any, Iterable

try:  # pragma: no cover - optional dependency in tests
    import streamlit as st
except Exception:  # pragma: no cover - Streamlit not available during some tests
    st = None  # type: ignore

DEBUG_RERUN = os.getenv("DEBUG_RERUN", "0") == "1"

_FALLBACK_EVENTS: deque[dict[str, Any]] = deque(maxlen=1024)
_FALLBACK_LOCK = threading.Lock()


def _append_event(event: dict[str, Any]) -> None:
    if not DEBUG_RERUN:
        return
    if st is not None:
        try:
            events = st.session_state.setdefault("_debug_events", [])
            events.append(event)
            return
        except Exception:
            # Fall back to in-memory storage when session_state is unavailable
            pass
    with _FALLBACK_LOCK:
        _FALLBACK_EVENTS.append(event)


def _current_flow_id() -> str | None:
    if st is None:
        return None
    try:
        flow_id = st.session_state.get("_ui_flow_id")
    except Exception:
        return None
    if isinstance(flow_id, str) and flow_id:
        return flow_id
    return None


def mark_event(kind: str, detail: str = "", extra: dict[str, Any] | None = None) -> None:
    """Record a debug event when rerun tracing is enabled."""

    if not DEBUG_RERUN:
        return
    payload: dict[str, Any] = {
        "ts": time.time(),
        "kind": kind,
        "detail": detail,
        "flow_id": _current_flow_id(),
        "thread": threading.current_thread().name,
    }
    if extra:
        payload.update(extra)
    _append_event(payload)


def _format_stack(frames: Iterable[str]) -> list[str]:
    return [frame.rstrip() for frame in frames]


def safe_rerun(
    reason: str,
    *,
    rerun_args: tuple[Any, ...] | None = None,
    rerun_kwargs: dict[str, Any] | None = None,
) -> None:
    """Invoke Streamlit's rerun primitive while tracing the request."""

    runner = None
    if st is not None:
        runner = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
    if runner is None:
        raise RuntimeError("Streamlit rerun primitive not available")

    mark_event(
        "rerun",
        reason,
        {
            "stack": _format_stack(traceback.format_stack(limit=6)),
            "args": list(rerun_args or ()),
            "kwargs": dict(rerun_kwargs or {}),
        },
    )

    args = rerun_args or ()
    kwargs = rerun_kwargs or {}
    try:
        runner(*args, **kwargs)
    except TypeError:
        # Some Streamlit versions reject positional arguments; retry without them.
        runner()


def safe_stop(reason: str = "") -> None:
    """Invoke ``st.stop`` capturing the reason in the debug timeline."""

    stopper = getattr(st, "stop", None) if st is not None else None
    if stopper is None:
        raise RuntimeError("Streamlit stop primitive not available")

    mark_event("stop", reason)
    stopper()


def get_fallback_events() -> list[dict[str, Any]]:
    """Return events captured when Streamlit's session state was unavailable."""

    with _FALLBACK_LOCK:
        return list(_FALLBACK_EVENTS)


__all__ = ["DEBUG_RERUN", "mark_event", "safe_rerun", "safe_stop", "get_fallback_events"]
