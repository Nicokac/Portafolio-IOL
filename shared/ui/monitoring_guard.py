"""Utilities to coordinate heavy workloads when monitoring panels are active."""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


_STATE_KEY = "_monitoring_active_panel"


def _is_valid_selection(candidate: Any) -> bool:
    if not isinstance(candidate, dict):
        return False
    module = candidate.get("module")
    attr = candidate.get("attr")
    return bool(module) and bool(attr)


def is_monitoring_active() -> bool:
    """Return ``True`` when an inline monitoring panel is active."""

    try:
        selection = st.session_state.get(_STATE_KEY)
    except Exception:  # pragma: no cover - defensive safeguard for missing state
        logger.debug("[monitoring] session_state inaccesible al consultar el guard", exc_info=True)
        return False

    if not _is_valid_selection(selection):
        return False

    return True


__all__ = ["is_monitoring_active"]
