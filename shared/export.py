from __future__ import annotations

import logging
from functools import wraps
from time import monotonic
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


@_wrap_with_cache
def _get_kaleido_scope():
    """Obtiene un alcance de kaleido cacheado para reutilizar el proceso de Chromium."""
    try:
        return pio.kaleido.scope
    except Exception as exc:  # pragma: no cover - depende de la instalación local
        _log_noncritical_export_failure(exc)
        raise


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
