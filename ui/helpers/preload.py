"""Utilities to gate scientific dashboards on preload completion."""

from __future__ import annotations

import importlib
import logging
from contextlib import nullcontext
from types import ModuleType
from typing import Any, Callable, ContextManager

try:  # pragma: no cover - defensive guard for optional dependency
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover - requires Streamlit runtime
    raise RuntimeError(
        "Streamlit no está instalado. Verificá que el entorno tenga el paquete "
        "`streamlit` declarado en requirements.txt antes de iniciar la app."
    ) from exc


def _resolve_streamlit_api_exception() -> type[BaseException]:
    """Return the Streamlit API exception class available in the runtime."""

    # Streamlit has moved the public exception class through different modules in
    # recent releases.  Instead of hard-coding a single import we try a list of
    # known locations and gracefully fall back to ``Exception`` when none of
    # them are available.  This keeps the preload helper compatible across
    # Streamlit versions without failing during module import time.
    candidate_paths = (
        "streamlit.errors.StreamlitAPIException",
        "streamlit.runtime.scriptrunner.script_runner.StreamlitAPIException",
        "streamlit.runtime.exceptions.StreamlitAPIException",
    )

    for dotted_path in candidate_paths:  # pragma: no branch - short candidate list
        module_path, _, attr_name = dotted_path.rpartition(".")
        if not module_path:
            continue
        try:
            module = importlib.import_module(module_path)
        except Exception:  # pragma: no cover - depends on optional Streamlit bits
            continue
        exc_type = getattr(module, attr_name, None)
        if isinstance(exc_type, type) and issubclass(exc_type, BaseException):
            return exc_type

    # ``StreamlitAPIException`` may not be present in minimal environments (for
    # example, running the API service without Streamlit installed).  In those
    # cases we only need an ``Exception`` subclass to keep the helper working.
    fallback = getattr(st, "StreamlitAPIException", None)
    if isinstance(fallback, type) and issubclass(fallback, BaseException):
        return fallback

    return Exception


StreamlitAPIException = _resolve_streamlit_api_exception()

_SESSION_KEY = "scientific_preload_ready"
_WORKER_MODULE = "services.preload_worker"
_LOGGER = logging.getLogger(__name__)
_MISSING = object()


def _resolve_placeholder(container: Any):
    if hasattr(container, "empty"):
        try:
            return container.empty()
        except StreamlitAPIException:  # pragma: no cover - requires missing context
            _LOGGER.debug(
                "[preload] No se pudo obtener placeholder del contenedor", exc_info=True
            )
        except Exception:  # pragma: no cover - defensive guard
            _LOGGER.debug(
                "[preload] Error inesperado al obtener placeholder del contenedor",
                exc_info=True,
            )
    try:
        return st.empty()
    except StreamlitAPIException:  # pragma: no cover - requires missing context
        _LOGGER.debug("[preload] st.empty() indisponible, se omite placeholder", exc_info=True)
        return None
    except Exception:  # pragma: no cover - defensive guard
        _LOGGER.debug(
            "[preload] Error inesperado al crear placeholder global", exc_info=True
        )
        return None


def _resolve_placeholder_cm(placeholder: Any) -> ContextManager[Any]:
    if placeholder is None:
        return nullcontext()
    container_method = getattr(placeholder, "container", None)
    if callable(container_method):
        try:
            return container_method()
        except StreamlitAPIException:  # pragma: no cover - requires missing context
            _LOGGER.debug(
                "[preload] Placeholder.container() falló, se omite el contexto",
                exc_info=True,
            )
        except Exception:  # pragma: no cover - defensive guard
            _LOGGER.debug(
                "[preload] Error inesperado al construir el contexto del placeholder",
                exc_info=True,
            )
    return nullcontext()


def _resolve_spinner(message: str) -> ContextManager[Any]:
    spinner = getattr(st, "spinner", None)
    if callable(spinner):
        try:
            return spinner(message)
        except StreamlitAPIException:  # pragma: no cover - requires missing context
            _LOGGER.debug(
                "[preload] Spinner de Streamlit indisponible, se omite", exc_info=True
            )
        except Exception:  # pragma: no cover - defensive guard
            _LOGGER.debug(
                "[preload] Error inesperado al crear spinner", exc_info=True
            )
    return nullcontext()


def _set_session_ready(value: bool) -> None:
    try:
        st.session_state[_SESSION_KEY] = value
    except StreamlitAPIException:  # pragma: no cover - requires missing context
        _LOGGER.debug(
            "[preload] session_state no disponible, se omite bandera", exc_info=True
        )
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
    placeholder_cm = _resolve_placeholder_cm(placeholder)
    spinner_cm = _resolve_spinner(message)
    try:
        with placeholder_cm:
            with spinner_cm:
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
            if placeholder is not None:
                placeholder.empty()
        except Exception:  # pragma: no cover - placeholder without empty method
            pass

    ready = _call_is_complete(is_complete)
    _set_session_ready(ready)
    return ready


__all__ = ["ensure_scientific_preload_ready"]
