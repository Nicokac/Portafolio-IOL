from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

try:  # pragma: no cover - streamlit puede no estar disponible en CLI
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - degradación a caché local
    st = None  # type: ignore

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_GLOBAL_CACHE: Dict[str, Any] = {}
_KALEIDO_RUNTIME_AVAILABLE: Optional[bool] = None
_KALEIDO_WARNING_EMITTED = False
_KALEIDO_WARNING_MESSAGE = (
    "📉 Exportación a PNG deshabilitada. Instale los requisitos de Kaleido/Chrome para"
    " habilitar las descargas."
)


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


def _log_kaleido_warning(exc: Exception | None = None) -> None:
    """Registra un único warning si kaleido no está disponible."""

    global _KALEIDO_WARNING_EMITTED

    if not _KALEIDO_WARNING_EMITTED:
        if exc is None:
            logger.warning(_KALEIDO_WARNING_MESSAGE)
        else:
            logger.warning("%s (%s)", _KALEIDO_WARNING_MESSAGE, exc)
        _KALEIDO_WARNING_EMITTED = True
    elif exc is not None:
        logger.debug("Kaleido runtime no disponible: %s", exc)


def _mark_runtime_unavailable(exc: Exception | None = None) -> None:
    """Actualiza el estado del runtime y emite el warning correspondiente."""

    global _KALEIDO_RUNTIME_AVAILABLE

    _KALEIDO_RUNTIME_AVAILABLE = False
    _log_kaleido_warning(exc)


@_wrap_with_cache
def _get_kaleido_scope():
    """Obtiene un alcance de kaleido cacheado para reutilizar el proceso de Chromium."""
    return pio.kaleido.scope


def ensure_kaleido_runtime() -> bool:
    """Garantiza que el runtime de Kaleido y Chromium esté disponible."""

    global _KALEIDO_RUNTIME_AVAILABLE

    if _KALEIDO_RUNTIME_AVAILABLE is not None:
        return _KALEIDO_RUNTIME_AVAILABLE

    try:
        scope = _get_kaleido_scope()
    except Exception as exc:  # pragma: no cover - depende de la instalación local
        _mark_runtime_unavailable(exc)
        return False

    ensure_chrome = getattr(scope, "ensure_chrome", None)

    if not callable(ensure_chrome):  # pragma: no cover - versiones antiguas no soportadas
        _mark_runtime_unavailable(RuntimeError("kaleido scope lacks ensure_chrome"))
        return False

    try:
        ensure_chrome()
    except (RuntimeError, OSError) as exc:
        _mark_runtime_unavailable(exc)
        return False

    _KALEIDO_RUNTIME_AVAILABLE = True
    return True


def fig_to_png_bytes(fig: go.Figure) -> Optional[bytes]:
    """Devuelve la figura renderizada como bytes PNG usando kaleido si está disponible."""

    if not ensure_kaleido_runtime():
        return None

    try:
        return pio.to_image(fig, format="png")
    except (RuntimeError, OSError) as exc:
        _mark_runtime_unavailable(exc)
        return None
