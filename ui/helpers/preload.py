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

    if worker.is_preload_complete():
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
            worker.wait_for_preload_completion()
    placeholder.empty()

    ready = bool(worker.is_preload_complete())
    try:
        st.session_state[_SESSION_KEY] = ready
    except Exception:
        pass
    return ready


__all__ = ["ensure_scientific_preload_ready"]
