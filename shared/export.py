from __future__ import annotations

import importlib
import importlib.util
import logging
import shutil
import threading
import warnings
from functools import wraps
from time import monotonic, perf_counter
from typing import Any, Callable, Dict, Optional, TypeVar

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from services.environment import record_kaleido_lazy_load

logger = logging.getLogger(__name__)

try:
    if getattr(pio.renderers, "default", None) != "browser":
        pio.renderers.default = "browser"
        logger.info("Plotly renderer fallback set to browser mode")
except Exception:  # pragma: no cover - defensive guard
    logger.debug("No se pudo configurar el renderer de Plotly", exc_info=True)

try:  # pragma: no cover - detection is lightweight
    _KALEIDO_AVAILABLE = importlib.util.find_spec("kaleido") is not None
except Exception:  # pragma: no cover - defensive guard
    _KALEIDO_AVAILABLE = False

if not _KALEIDO_AVAILABLE:
    logger.warning("⚠️ Dependencia Kaleido no instalada — export a imagen deshabilitado")

_KALEIDO_IMPORTED = False
_KALEIDO_IMPORT_LOCK = threading.Lock()

try:  # pragma: no cover - streamlit puede no estar disponible en CLI
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - degradación a caché local
    st = None  # type: ignore

_T = TypeVar("_T")
_GLOBAL_CACHE: Dict[str, Any] = {}
_KALEIDO_RUNTIME_AVAILABLE: Optional[bool] = None
_KALEIDO_WARNING_LAST_TS: Optional[float] = None
_KALEIDO_WARNING_WINDOW_SECONDS = 300


def _wrap_with_cache(func: Callable[[], _T]) -> Callable[[], _T]:
    """Intenta aplicar `st.cache_resource` y degrada a un caché global."""

    def _dict_cache() -> Callable[[], _T]:
        @wraps(func)
        def wrapper() -> _T:
            if func.__name__ not in _GLOBAL_CACHE:
                _GLOBAL_CACHE[func.__name__] = func()
            return _GLOBAL_CACHE[func.__name__]

        return wrapper

    cache_resource = getattr(st, "cache_resource", None) if st is not None else None
    if cache_resource is None:
        return _dict_cache()

    try:
        return cache_resource(func)
    except Exception:  # pragma: no cover - fallback ante errores de Streamlit
        return _dict_cache()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Devuelve un DataFrame codificado como bytes CSV UTF-8 sin índice."""
    return df.to_csv(index=False).encode("utf-8")


def _log_noncritical_export_failure(exc: Exception) -> None:
    """Registra advertencias amortiguadas para fallas no críticas."""

    global _KALEIDO_WARNING_LAST_TS

    now = monotonic()
    last_ts = _KALEIDO_WARNING_LAST_TS

    if last_ts is None or (now - last_ts) >= _KALEIDO_WARNING_WINDOW_SECONDS:
        logger.warning(
            "⚠️ Exportación no crítica fallida (Kaleido)",
            exc_info=exc,
        )
        _KALEIDO_WARNING_LAST_TS = now


def _mark_runtime_unavailable(exc: Exception | None = None) -> None:
    """Actualiza el estado del runtime para evitar reintentos redundantes."""

    global _KALEIDO_RUNTIME_AVAILABLE

    _KALEIDO_RUNTIME_AVAILABLE = False


def _lazy_import_kaleido() -> bool:
    """Importa Kaleido bajo demanda registrando métricas de latencia."""

    global _KALEIDO_AVAILABLE, _KALEIDO_IMPORTED, _KALEIDO_RUNTIME_AVAILABLE

    if not _KALEIDO_AVAILABLE:
        _KALEIDO_RUNTIME_AVAILABLE = False
        return False

    if _KALEIDO_IMPORTED:
        return True

    with _KALEIDO_IMPORT_LOCK:
        if _KALEIDO_IMPORTED:
            return True

        start = perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                    module=r"plotly\.io",
                )
                warnings.filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                    module=r"kaleido\.scopes",
                )
                importlib.import_module("kaleido")
                plotly_io = importlib.import_module("plotly.io")
                getattr(plotly_io, "write_image", None)
        except Exception as exc:  # pragma: no cover - depende del entorno
            _KALEIDO_AVAILABLE = False
            _KALEIDO_RUNTIME_AVAILABLE = False
            _KALEIDO_IMPORTED = False
            _mark_runtime_unavailable(exc)
            logger.warning(
                "⚠️ Kaleido deshabilitado (%s) — exportación a imagen omitida",
                exc,
            )
            return False

        end = perf_counter()
        duration_ms = (end - start) * 1000.0
        _KALEIDO_IMPORTED = True
        _KALEIDO_RUNTIME_AVAILABLE = None
        record_kaleido_lazy_load(duration_ms, completed_at=end)

    if shutil.which("chromium") is None:
        logger.warning(
            "⚠️ Kaleido detectado pero sin Chromium disponible — export limitada"
        )

    return True


@_wrap_with_cache
def _get_kaleido_scope():
    """Obtiene un alcance de kaleido cacheado para reutilizar el proceso de Chromium."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module=r"plotly\.io",
            )
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module=r"kaleido\.scopes",
            )
            return pio.kaleido.scope
    except Exception as exc:  # pragma: no cover - depende de la instalación local
        _log_noncritical_export_failure(exc)
        raise


def ensure_kaleido_runtime() -> bool:
    """Garantiza que el runtime de Kaleido y Chromium esté disponible."""

    global _KALEIDO_RUNTIME_AVAILABLE

    if not _lazy_import_kaleido():
        return False

    if _KALEIDO_RUNTIME_AVAILABLE is not None:
        return _KALEIDO_RUNTIME_AVAILABLE

    try:
        scope = _get_kaleido_scope()
    except Exception as exc:  # pragma: no cover - depende de la instalación local
        _mark_runtime_unavailable(exc)
        return False

    ensure_chrome = getattr(scope, "ensure_chrome", None)

    if not callable(ensure_chrome):  # pragma: no cover - versiones antiguas no soportadas
        exc = RuntimeError("kaleido scope lacks ensure_chrome")
        _log_noncritical_export_failure(exc)
        _mark_runtime_unavailable(exc)
        return False

    try:
        ensure_chrome()
    except Exception as exc:
        _log_noncritical_export_failure(exc)
        _mark_runtime_unavailable(exc)
        return False

    _KALEIDO_RUNTIME_AVAILABLE = True
    return True


def fig_to_png_bytes(fig: go.Figure) -> Optional[bytes]:
    """Devuelve la figura renderizada como bytes PNG usando kaleido si está disponible."""

    if not _lazy_import_kaleido():
        logger.warning("⚠️ Exportación no crítica omitida (Kaleido no disponible)")
        return None

    if ensure_kaleido_runtime():
        try:
            return pio.to_image(fig, format="png")
        except Exception as exc:
            _log_noncritical_export_failure(exc)
            _mark_runtime_unavailable(exc)
            return None

    # Intento de gracia para entornos donde ensure_chrome no está disponible pero kaleido puede funcionar.
    try:  # pragma: no cover - ejercido mediante stubs en pruebas de UI
        return pio.to_image(fig, format="png")
    except Exception as exc:  # pragma: no cover - se registra como falla no crítica
        _log_noncritical_export_failure(exc)
        return None
