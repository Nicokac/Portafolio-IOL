from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, Dict, TypeVar

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


@_wrap_with_cache
def _get_kaleido_scope():
    """Obtiene un alcance de kaleido cacheado para reutilizar el proceso de Chromium."""
    return pio.kaleido.scope


def fig_to_png_bytes(fig: go.Figure) -> bytes:
    """Devuelve la figura renderizada como bytes PNG usando kaleido."""
    try:
        _get_kaleido_scope()
        return pio.to_image(fig, format="png")
    except (ValueError, RuntimeError, TypeError) as e:  # pragma: no cover - depende de librerías externas
        logger.exception("kaleido no disponible")
        raise ValueError("kaleido no disponible") from e
