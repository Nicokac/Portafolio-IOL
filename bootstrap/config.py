"""Application bootstrap configuration helpers."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from uuid import uuid4

def _is_headless_mode() -> bool:
    """Return True when running under the light-weight UNIT_TEST profile."""

    return os.environ.get("UNIT_TEST") == "1"


def _load_streamlit_module() -> ModuleType:
    """Load the Streamlit module or a minimal stub for headless tests."""

    if not _is_headless_mode():
        import streamlit as real_streamlit

        return real_streamlit

    stub = ModuleType("streamlit")
    stub.session_state = {}
    stub.secrets = {}
    noop = lambda *_, **__: None  # noqa: E731 - simple stub helpers
    stub.warning = noop
    stub.info = noop
    stub.caption = noop
    stub.write = noop
    stub.divider = noop
    stub.plotly_chart = noop
    stub.selectbox = lambda *_, **__: None
    stub.subheader = noop
    stub.dataframe = noop
    stub.table = noop
    stub.container = lambda *_, **__: nullcontext()
    stub.columns = lambda spec=None, *_, **__: tuple(nullcontext() for _ in range((spec or 2) if isinstance(spec, int) else 2))
    stub.set_page_config = noop
    stub.markdown = noop
    stub.sidebar = SimpleNamespace(markdown=noop, radio=noop)
    stub.stop = lambda: None
    stub.__getattr__ = lambda _name: noop
    sys.modules["streamlit"] = stub
    return stub


st = _load_streamlit_module()

from shared import skeletons  # noqa: E402 - ensure Streamlit stub is registered first

logger = logging.getLogger(__name__)


TOTAL_LOAD_START = time.perf_counter()
if not _is_headless_mode() and skeletons.initialize(TOTAL_LOAD_START):
    logger.info("🧩 Skeleton system initialized at %.2f", TOTAL_LOAD_START)


def _patch_streamlit_runtime() -> None:
    """Provide compatibility helpers for environments with limited Streamlit APIs."""

    if not hasattr(st, "stop"):
        setattr(st, "stop", lambda: None)

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

    from services.startup_logger import log_startup_event
    from shared.config import configure_logging, ensure_tokens_key
    from shared.qa_profiler import record_startup_complete
    from shared.security_env_validator import validate_security_environment
    from shared.version import __build_signature__, __version__

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
    validation = validate_security_environment()
    if validation.relaxed:
        missing_items = "\n- " + "\n- ".join(validation.errors) if validation.errors else ""
        guidance = (
            "Faltan claves obligatorias (FASTAPI_TOKENS_KEY / IOL_TOKENS_KEY). "
            "Generá nuevas con `python generate_key.py` y exportalas en tu entorno."
        )
        warning_msg = (
            "Modo relajado habilitado en Streamlit (APP_ENV=%s). %s%s"
            % (validation.app_env or "desconocido", guidance, missing_items)
        )
        logger.warning(warning_msg)
        try:
            st.warning(warning_msg)
        except Exception:  # pragma: no cover - defensive UI hook
            logger.debug("No se pudo mostrar el aviso de claves faltantes en la UI.")

    if _is_headless_mode():
        logger.info("UNIT_TEST=1 detectado — inicialización de UI omitida.")
    else:
        from ui.ui_settings import init_ui

        init_ui()

    message = f"App initialized — version={__version__} — build={__build_signature__}"
    log_startup_event(message)
    logger.info(message)

    try:
        record_startup_complete()
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("Unable to record QA startup metric", exc_info=True)

    return args


__all__ = ["TOTAL_LOAD_START", "init_app"]
