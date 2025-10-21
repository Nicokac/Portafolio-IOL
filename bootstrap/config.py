"""Application bootstrap configuration helpers."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from contextlib import nullcontext
from uuid import uuid4

import streamlit as st

from services.startup_logger import log_startup_event
from shared import skeletons
from shared.config import configure_logging, ensure_tokens_key
from shared.security_env_validator import validate_security_environment
from shared.version import __build_signature__, __version__
from ui.ui_settings import init_ui

logger = logging.getLogger(__name__)


TOTAL_LOAD_START = time.perf_counter()
if skeletons.initialize(TOTAL_LOAD_START):
    logger.info("🧩 Skeleton system initialized at %.2f", TOTAL_LOAD_START)


def _patch_streamlit_runtime() -> None:
    """Provide compatibility helpers for environments with limited Streamlit APIs."""

    if not hasattr(st, "stop"):
        st.stop = lambda: None  # type: ignore[attr-defined]

    if not hasattr(st, "container"):
        st.container = lambda *_, **__: nullcontext()  # type: ignore[attr-defined]

    if not hasattr(st, "columns"):
        def _dummy_columns(spec: Sequence[int] | int | None = None, *args, **kwargs):
            if isinstance(spec, int):
                count = max(spec, 1)
            elif isinstance(spec, Sequence):
                count = max(len(spec), 1)
            else:
                count = 2
            return tuple(nullcontext() for _ in range(count))

        st.columns = _dummy_columns  # type: ignore[attr-defined]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments relevant for logging."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--log-level", dest="log_level")
    parser.add_argument("--log-format", dest="log_format", choices=["text", "json"])
    args, _ = parser.parse_known_args(argv)
    return args


def init_app(argv: list[str] | None = None) -> argparse.Namespace:
    """Bootstrap logging, environment validation and Streamlit session state."""

    _patch_streamlit_runtime()

    args = _parse_args(argv or [])
    configure_logging(
        level=args.log_level,
        json_format=(args.log_format == "json") if args.log_format else None,
    )

    log_startup_event("Streamlit app bootstrap initiated")

    logger.info("requirements.txt es la fuente autorizada de dependencias.")

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid4().hex

    ensure_tokens_key()
    validate_security_environment()
    init_ui()

    message = (
        f"App initialized — version={__version__} — build={__build_signature__}"
    )
    log_startup_event(message)
    logger.info(message)

    return args


__all__ = ["TOTAL_LOAD_START", "init_app"]
