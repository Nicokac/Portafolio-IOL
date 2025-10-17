"""Utilities to track when the first visual placeholder becomes visible."""

from __future__ import annotations

import logging
import time
from typing import Any, Tuple

import streamlit as st

_LOGGER = logging.getLogger(__name__)

_START_KEY = "_ui_skeleton_start"
_METRIC_KEY = "_ui_skeleton_render_ms"
_LABEL_KEY = "_ui_skeleton_label"

_FALLBACK_START = time.perf_counter()
_FALLBACK_METRIC: float | None = None
_FALLBACK_LABEL: str | None = None


def initialize(start: float | None = None) -> None:
    """Persist the reference start timestamp used for skeleton latency."""

    global _FALLBACK_START

    try:
        if start is None:
            start = float(time.perf_counter())
        else:
            start = float(start)
    except (TypeError, ValueError):
        start = float(time.perf_counter())
    _FALLBACK_START = start
    try:
        st.session_state.setdefault(_START_KEY, start)
    except Exception:  # pragma: no cover - session state may be read-only
        pass


def _get_start() -> float:
    try:
        value = st.session_state.get(_START_KEY)
    except Exception:  # pragma: no cover - session state may be read-only
        value = None
    if isinstance(value, (int, float)):
        return float(value)
    return float(_FALLBACK_START)


def _is_recorded() -> bool:
    try:
        recorded = st.session_state.get(_METRIC_KEY)
        if recorded is not None:
            return True
    except Exception:  # pragma: no cover - session state may be read-only
        pass
    return _FALLBACK_METRIC is not None


def _resolve_placeholder_container(placeholder: Any) -> Any | None:
    """Return a safe container for the provided placeholder if possible."""

    if placeholder is None:
        return None
    container = None
    try:
        container_fn = getattr(placeholder, "container", None)
        if callable(container_fn):
            container = container_fn()
    except Exception:  # pragma: no cover - defensive guard for Streamlit stubs
        _LOGGER.debug("No se pudo obtener el contenedor del placeholder", exc_info=True)
        container = None
    if container is None:
        container = placeholder
    return container


def mark_placeholder(label: str, *, placeholder: Any | None = None) -> Any | None:
    """Register the first placeholder render latency if not recorded yet."""

    global _FALLBACK_METRIC, _FALLBACK_LABEL

    _LOGGER.info("🧩 Skeleton render called for %s", label)

    if not _is_recorded():
        elapsed = max((time.perf_counter() - _get_start()) * 1000.0, 0.0)
        try:
            st.session_state[_METRIC_KEY] = elapsed
            st.session_state[_LABEL_KEY] = label
        except Exception:  # pragma: no cover - session state may be read-only
            _FALLBACK_METRIC = elapsed
            _FALLBACK_LABEL = label
        else:
            _FALLBACK_METRIC = elapsed
            _FALLBACK_LABEL = label
        _LOGGER.debug("Skeleton placeholder recorded for %s at %.2f ms", label, elapsed)

    return _resolve_placeholder_container(placeholder)


def get_metric() -> Tuple[float | None, str | None]:
    """Return the recorded skeleton latency and associated label if available."""

    try:
        metric = st.session_state.get(_METRIC_KEY)
        label = st.session_state.get(_LABEL_KEY)
    except Exception:  # pragma: no cover - session state may be read-only
        metric = None
        label = None
    if metric is None:
        metric = _FALLBACK_METRIC
    if label is None:
        label = _FALLBACK_LABEL
    if isinstance(metric, (int, float)):
        metric = float(metric)
    else:
        metric = None
    if isinstance(label, str):
        return metric, label
    return metric, None


__all__ = ["initialize", "mark_placeholder", "get_metric"]
