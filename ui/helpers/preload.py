"""Utilities to gate scientific dashboards on preload completion."""

from __future__ import annotations

import importlib
from typing import Any

import streamlit as st

_SESSION_KEY = "scientific_preload_ready"


def _resolve_placeholder(container: Any):
    if hasattr(container, "empty"):
        return container.empty()
    return st.empty()


def ensure_scientific_preload_ready(
    container: Any,
    *,
    message: str = "Cargando librerías científicas…",
) -> bool:
    """Block the UI with a spinner until the preload worker finishes."""

    worker = importlib.import_module("services.preload_worker")

    is_complete = getattr(worker, "is_preload_complete", None)
    if is_complete is None:
        # Fallback for environments running an older preload worker implementation
        # where the readiness check is not available. In that scenario we assume
        # the worker finished so the UI can continue operating.
        try:
            st.session_state[_SESSION_KEY] = True
        except Exception:
            pass
        return True

    if is_complete():
        try:
            st.session_state[_SESSION_KEY] = True
        except Exception:
            pass
        return True

    try:
        st.session_state[_SESSION_KEY] = False
    except Exception:
        pass

    placeholder = _resolve_placeholder(container)
    with placeholder.container():
        with st.spinner(message):
            wait_for_completion = getattr(worker, "wait_for_preload_completion", None)
            if callable(wait_for_completion):
                wait_for_completion()
    placeholder.empty()

    ready = bool(is_complete())
    try:
        st.session_state[_SESSION_KEY] = ready
    except Exception:
        pass
    return ready


__all__ = ["ensure_scientific_preload_ready"]
