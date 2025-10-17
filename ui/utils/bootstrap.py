"""Helpers to load shared UI assets for Streamlit."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import streamlit as st

_LOGGER = logging.getLogger(__name__)


_BOOTSTRAP_FLAG = "_ui_bootstrap_css_loaded"


def _bootstrap_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / ".streamlit" / "bootstrap.css"


@lru_cache(maxsize=1)
def _read_bootstrap_css() -> str:
    path = _bootstrap_path()
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _LOGGER.warning("Bootstrap CSS file not found at %s", path)
    except OSError:
        _LOGGER.exception("Unable to read bootstrap CSS from %s", path)
    return ""


def ensure_bootstrap_assets() -> None:
    """Inject the shared CSS bundle once per Streamlit session."""

    css = _read_bootstrap_css()
    if not css:
        return

    try:
        if st.session_state.get(_BOOTSTRAP_FLAG):
            return
    except Exception:  # pragma: no cover - session state may be read-only
        pass

    try:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception:  # pragma: no cover - defensive guard for Streamlit stubs
        _LOGGER.debug("Unable to inject bootstrap CSS", exc_info=True)
        return

    try:
        st.session_state[_BOOTSTRAP_FLAG] = True
    except Exception:  # pragma: no cover - session state may be read-only
        pass


__all__ = ["ensure_bootstrap_assets"]
