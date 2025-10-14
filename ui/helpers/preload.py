"""Utilities to gate scientific dashboards on preload completion."""

from __future__ import annotations

import importlib
import logging
from types import ModuleType
from typing import Any, Callable

import streamlit as st

_SESSION_KEY = "scientific_preload_ready"
_WORKER_MODULE = "services.preload_worker"
_LOGGER = logging.getLogger(__name__)
_MISSING = object()


def _resolve_placeholder(container: Any):
    if hasattr(container, "empty"):
        return container.empty()
    return st.empty()


def _set_session_ready(value: bool) -> None:
    try:
        st.session_state[_SESSION_KEY] = value
    except Exception:  # pragma: no cover - safety when Streamlit state unavailable
        pass


def _import_preload_worker() -> ModuleType | None:
    try:
        return importlib.import_module(_WORKER_MODULE)
    except Exception:
        _LOGGER.warning(
            "[preload] No se pudo importar %s, se omite la espera de precarga",
            _WORKER_MODULE,
            exc_info=True,
        )
        return None


def _resolve_worker_api(
    worker: ModuleType,
) -> tuple[Callable[[], bool] | None, Callable[[float | None], bool] | None, bool]:
    is_complete = getattr(worker, "is_preload_complete", _MISSING)
    wait_for_completion = getattr(worker, "wait_for_preload_completion", None)
    is_partial_module = False

    if is_complete is _MISSING or not callable(is_complete):
        available = [name for name in dir(worker) if not name.startswith("_")]
        _LOGGER.warning(
            "[preload] El módulo %s está incompleto; falta is_preload_complete. "
            "Atributos disponibles: %s",
            _WORKER_MODULE,
            ", ".join(sorted(available)) or "<ninguno>",
        )
        setattr(worker, "is_preload_complete", lambda: False)
        is_partial_module = True
        is_complete = getattr(worker, "is_preload_complete")

    if wait_for_completion is None or not callable(wait_for_completion):
        _LOGGER.warning(
            "[preload] El módulo %s no expone wait_for_preload_completion válido",
            _WORKER_MODULE,
        )
        wait_for_completion = None

    return is_complete, wait_for_completion, is_partial_module


def _call_is_complete(func: Callable[[], bool]) -> bool:
    try:
        return bool(func())
    except Exception:
        _LOGGER.warning(
            "[preload] Error al consultar is_preload_complete, se asume False",
            exc_info=True,
        )
        return False


def ensure_scientific_preload_ready(
    container: Any,
    *,
    message: str = "Cargando librerías científicas…",
) -> bool:
    """Block the UI with a spinner until the preload worker finishes."""

    worker = _import_preload_worker()
    if worker is None:
        _set_session_ready(False)
        return False

    is_complete, wait_for_completion, is_partial_module = _resolve_worker_api(worker)
    if is_partial_module:
        _set_session_ready(False)
        return False

    if is_complete is None:
        _set_session_ready(False)
        return False

    if _call_is_complete(is_complete):
        _set_session_ready(True)
        return True

    _set_session_ready(False)

    if wait_for_completion is None:
        return False

    placeholder = _resolve_placeholder(container)
    try:
        with placeholder.container():
            with st.spinner(message):
                try:
                    wait_for_completion(None)
                except TypeError:
                    # Some implementations may not accept the timeout parameter.
                    try:
                        wait_for_completion()
                    except Exception:
                        _LOGGER.warning(
                            "[preload] Error al esperar la precarga científica",
                            exc_info=True,
                        )
                        return False
                except Exception:
                    _LOGGER.warning(
                        "[preload] Error al esperar la precarga científica",
                        exc_info=True,
                    )
                    return False
    finally:
        try:
            placeholder.empty()
        except Exception:  # pragma: no cover - placeholder without empty method
            pass

    ready = _call_is_complete(is_complete)
    _set_session_ready(ready)
    return ready


__all__ = ["ensure_scientific_preload_ready"]
