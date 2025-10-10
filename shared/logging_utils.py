"""Centralised logging helpers for Streamlit integrations."""

from __future__ import annotations

import logging
import os
import warnings
from typing import Iterable

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

_STREAMLIT_LOGGERS: tuple[str, ...] = (
    "streamlit.runtime.scriptrunner_utils.script_run_context",
    "streamlit.runtime.state.session_state",
    "streamlit.runtime.scriptrunner_utils",
    "streamlit",
)

_WARNING_FILTERS: tuple[tuple[str, str, str], ...] = (
    ("ignore", r".*bare mode.*", "streamlit"),
    ("ignore", r".*use_container_width.*", "streamlit"),
)


def configure_default_logging(level: int = logging.INFO) -> None:
    """Initialise the default logging configuration for the app."""

    logging.basicConfig(level=level, format=_DEFAULT_FORMAT)


def silence_streamlit_warnings(extra_loggers: Iterable[str] | None = None) -> None:
    """Suppress noisy Streamlit logging and known warnings."""

    targets = list(_STREAMLIT_LOGGERS)
    if extra_loggers:
        targets.extend(str(name) for name in extra_loggers)

    os.environ.setdefault("STREAMLIT_SUPPRESS_USE_CONTAINER_WIDTH_WARNING", "1")

    for name in targets:
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False

    for action, message, module in _WARNING_FILTERS:
        warnings.filterwarnings(action, message, module=module)

    try:
        import streamlit as _st  # Local import to avoid mandatory dependency on import time
    except Exception:
        return

    if not hasattr(_st, "rerun"):
        _st.rerun = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    try:
        _st.set_option("global.deprecation.use_container_width", False)
    except Exception:
        pass


__all__ = ["configure_default_logging", "silence_streamlit_warnings"]
