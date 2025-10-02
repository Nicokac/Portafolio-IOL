from __future__ import annotations

"""Helpers to capture health metrics and expose them via ``st.session_state``."""

from typing import Any, Dict, Iterable, Mapping, Optional
import time

import streamlit as st


_HEALTH_KEY = "health_metrics"


def _store() -> Dict[str, Any]:
    """Return the mutable health metrics store from the session state."""
    return st.session_state.setdefault(_HEALTH_KEY, {})


def _clean_detail(detail: Optional[str]) -> Optional[str]:
    if detail is None:
        return None
    text = str(detail).strip()
    return text or None


def record_iol_refresh(success: bool, *, detail: Optional[str] = None) -> None:
    """Persist the outcome of the last IOL login/refresh attempt."""
    store = _store()
    store["iol_refresh"] = {
        "status": "success" if success else "error",
        "detail": _clean_detail(detail),
        "ts": time.time(),
    }


def record_yfinance_usage(source: str, *, detail: Optional[str] = None) -> None:
    """Persist whether Yahoo Finance or a fallback served the last request."""
    store = _store()
    store["yfinance"] = {
        "source": source,
        "detail": _clean_detail(detail),
        "ts": time.time(),
    }


def record_fx_api_response(
    *, error: Optional[str] = None, elapsed_ms: Optional[float] = None
) -> None:
    """Persist metadata about the latest FX API call."""
    store = _store()
    store["fx_api"] = {
        "status": "success" if not error else "error",
        "error": _clean_detail(error),
        "elapsed_ms": float(elapsed_ms) if elapsed_ms is not None else None,
        "ts": time.time(),
    }


def record_fx_cache_usage(mode: str, *, age: Optional[float] = None) -> None:
    """Persist information about session cache usage for FX rates."""
    store = _store()
    store["fx_cache"] = {
        "mode": mode,
        "age": float(age) if age is not None else None,
        "ts": time.time(),
    }


def record_portfolio_load(
    elapsed_ms: Optional[float], *, source: str, detail: Optional[str] = None
) -> None:
    """Persist response time and source for the latest portfolio load."""
    store = _store()
    store["portfolio"] = {
        "elapsed_ms": float(elapsed_ms) if elapsed_ms is not None else None,
        "source": source,
        "detail": _clean_detail(detail),
        "ts": time.time(),
    }


def record_quote_load(
    elapsed_ms: Optional[float], *, source: str, count: Optional[int] = None
) -> None:
    """Persist response time and source for the latest quote load."""
    store = _store()
    store["quotes"] = {
        "elapsed_ms": float(elapsed_ms) if elapsed_ms is not None else None,
        "source": source,
        "count": int(count) if count is not None else None,
        "ts": time.time(),
    }


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_sectors(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        item = value.strip()
        return [item] if item else None
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or None
    return None


def _normalize_origin_counts(value: Any) -> Optional[Dict[str, int | float]]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return None

    normalized: Dict[str, int | float] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        if not key:
            continue
        numeric = _as_optional_float(raw_value)
        if numeric is None:
            continue
        if float(numeric).is_integer():
            normalized[key] = int(numeric)
        else:
            normalized[key] = float(numeric)
    return normalized or None


def record_opportunities_report(
    *,
    mode: str,
    elapsed_ms: Optional[float],
    cached_elapsed_ms: Optional[float],
    universe_initial: Optional[Any] = None,
    universe_final: Optional[Any] = None,
    discard_ratio: Optional[Any] = None,
    highlighted_sectors: Optional[Any] = None,
    counts_by_origin: Optional[Any] = None,
) -> None:
    """Persist cache usage metrics for the opportunities screening."""

    store = _store()
    entry: Dict[str, Any] = {
        "mode": mode,
        "elapsed_ms": float(elapsed_ms) if elapsed_ms is not None else None,
        "cached_elapsed_ms": (
            float(cached_elapsed_ms) if cached_elapsed_ms is not None else None
        ),
        "ts": time.time(),
    }

    initial = _as_optional_int(universe_initial)
    final = _as_optional_int(universe_final)
    if initial is not None:
        entry["universe_initial"] = initial
    if final is not None:
        entry["universe_final"] = final

    ratio = _as_optional_float(discard_ratio)
    if ratio is not None:
        entry["discard_ratio"] = ratio

    sectors = _normalize_sectors(highlighted_sectors)
    if sectors is not None:
        entry["highlighted_sectors"] = sectors

    origins = _normalize_origin_counts(counts_by_origin)
    if origins is not None:
        entry["counts_by_origin"] = origins

    store["opportunities"] = entry


def get_health_metrics() -> Dict[str, Any]:
    """Return a shallow copy of the tracked metrics for UI consumption."""
    store = _store()
    return {
        "iol_refresh": store.get("iol_refresh"),
        "yfinance": store.get("yfinance"),
        "fx_api": store.get("fx_api"),
        "fx_cache": store.get("fx_cache"),
        "portfolio": store.get("portfolio"),
        "quotes": store.get("quotes"),
        "opportunities": store.get("opportunities"),
    }


__all__ = [
    "get_health_metrics",
    "record_fx_api_response",
    "record_fx_cache_usage",
    "record_iol_refresh",
    "record_portfolio_load",
    "record_quote_load",
    "record_opportunities_report",
    "record_yfinance_usage",
]
