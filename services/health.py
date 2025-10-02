from __future__ import annotations

"""Helpers to capture health metrics and expose them via ``st.session_state``."""

from collections import deque
import math
from typing import Any, Deque, Dict, Iterable, Mapping, Optional
import time

import streamlit as st


_HEALTH_KEY = "health_metrics"
_OPPORTUNITIES_HISTORY_KEY = "opportunities_history"
_OPPORTUNITIES_STATS_KEY = "opportunities_stats"
_OPPORTUNITIES_HISTORY_LIMIT = 5


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


def record_macro_api_usage(
    *,
    provider: str,
    status: str,
    elapsed_ms: Optional[float] = None,
    detail: Optional[str] = None,
    fallback: bool = False,
) -> None:
    """Persist information about the macro/sector data provider."""

    store = _store()
    store["macro_api"] = {
        "provider": str(provider or "unknown"),
        "status": str(status or "unknown"),
        "elapsed_ms": float(elapsed_ms) if elapsed_ms is not None else None,
        "detail": _clean_detail(detail),
        "fallback": bool(fallback),
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
    **extra_metrics: Any,
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

    if extra_metrics:
        extras: Dict[str, Any] = {}
        for key, value in extra_metrics.items():
            normalized_key = str(key)
            # ``None`` values offer no insight and would just clutter the payload.
            if value is None:
                continue
            if isinstance(value, (list, tuple, set, frozenset)):
                extras[normalized_key] = [
                    item for item in value if item is not None
                ]
            elif isinstance(value, dict):
                extras[normalized_key] = {
                    str(sub_key): sub_value
                    for sub_key, sub_value in value.items()
                    if sub_value is not None
                }
            else:
                extras[normalized_key] = value
        if extras:
            entry.setdefault("extra_metrics", {}).update(extras)

    store["opportunities"] = entry

    stats = store.get(_OPPORTUNITIES_STATS_KEY)
    if not isinstance(stats, dict):
        stats = {
            "modes": {},
            "elapsed_sum": 0.0,
            "elapsed_sum_sq": 0.0,
            "elapsed_count": 0,
            "cached_sum": 0.0,
            "cached_sum_sq": 0.0,
            "cached_count": 0,
            "improvement_count": 0,
            "improvement_wins": 0,
            "improvement_losses": 0,
            "improvement_ties": 0,
            "improvement_delta_sum": 0.0,
        }

    modes: Dict[str, int] = stats.setdefault("modes", {})
    modes[mode] = int(modes.get(mode, 0)) + 1

    elapsed_value = _as_optional_float(elapsed_ms)
    if elapsed_value is not None:
        stats["elapsed_count"] = int(stats.get("elapsed_count", 0)) + 1
        stats["elapsed_sum"] = float(stats.get("elapsed_sum", 0.0)) + elapsed_value
        stats["elapsed_sum_sq"] = (
            float(stats.get("elapsed_sum_sq", 0.0)) + elapsed_value * elapsed_value
        )

    cached_value = _as_optional_float(cached_elapsed_ms)
    if cached_value is not None:
        stats["cached_count"] = int(stats.get("cached_count", 0)) + 1
        stats["cached_sum"] = float(stats.get("cached_sum", 0.0)) + cached_value
        stats["cached_sum_sq"] = (
            float(stats.get("cached_sum_sq", 0.0)) + cached_value * cached_value
        )

    if elapsed_value is not None and cached_value is not None:
        stats["improvement_count"] = int(stats.get("improvement_count", 0)) + 1
        diff = cached_value - elapsed_value
        stats["improvement_delta_sum"] = float(
            stats.get("improvement_delta_sum", 0.0)
        ) + diff
        if diff > 0:
            stats["improvement_wins"] = int(stats.get("improvement_wins", 0)) + 1
        elif diff < 0:
            stats["improvement_losses"] = int(stats.get("improvement_losses", 0)) + 1
        else:
            stats["improvement_ties"] = int(stats.get("improvement_ties", 0)) + 1

    store[_OPPORTUNITIES_STATS_KEY] = stats

    raw_history = store.get(_OPPORTUNITIES_HISTORY_KEY)
    history: Deque[Dict[str, Any]]
    if isinstance(raw_history, deque):
        history = raw_history
    elif isinstance(raw_history, list):
        history = deque(raw_history, maxlen=_OPPORTUNITIES_HISTORY_LIMIT)
    else:
        history = deque(maxlen=_OPPORTUNITIES_HISTORY_LIMIT)

    history.append(entry)
    store[_OPPORTUNITIES_HISTORY_KEY] = history


def get_health_metrics() -> Dict[str, Any]:
    """Return a shallow copy of the tracked metrics for UI consumption."""
    store = _store()

    def _summarize_stats(raw_stats: Any) -> Dict[str, Any]:
        if not isinstance(raw_stats, Mapping):
            return {}

        summary: Dict[str, Any] = {}

        modes: Dict[str, int] = {}
        raw_modes = raw_stats.get("modes")
        if isinstance(raw_modes, Mapping):
            for key, value in raw_modes.items():
                try:
                    modes[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
        if modes:
            total_modes = sum(max(count, 0) for count in modes.values())
            total_modes = max(total_modes, 0)
            if total_modes:
                summary["mode_counts"] = modes
                summary["mode_total"] = total_modes
                summary["mode_ratios"] = {
                    name: count / total_modes for name, count in modes.items()
                }
                hit_count = modes.get("hit", 0)
                summary["hit_ratio"] = hit_count / total_modes

        def _summarize_timing(prefix: str) -> Optional[Dict[str, Any]]:
            count = int(raw_stats.get(f"{prefix}_count", 0) or 0)
            if count <= 0:
                return None
            sum_value = float(raw_stats.get(f"{prefix}_sum", 0.0) or 0.0)
            sum_sq_value = float(raw_stats.get(f"{prefix}_sum_sq", 0.0) or 0.0)
            avg = sum_value / count
            variance = max(sum_sq_value / count - avg * avg, 0.0)
            stdev = math.sqrt(variance)
            return {"count": count, "avg": avg, "stdev": stdev}

        elapsed_stats = _summarize_timing("elapsed")
        if elapsed_stats:
            summary["elapsed"] = elapsed_stats

        cached_stats = _summarize_timing("cached")
        if cached_stats:
            summary["cached_elapsed"] = cached_stats

        improvement_count = int(raw_stats.get("improvement_count", 0) or 0)
        if improvement_count > 0:
            wins = int(raw_stats.get("improvement_wins", 0) or 0)
            losses = int(raw_stats.get("improvement_losses", 0) or 0)
            ties = int(raw_stats.get("improvement_ties", 0) or 0)
            delta_sum = float(raw_stats.get("improvement_delta_sum", 0.0) or 0.0)
            summary["improvement"] = {
                "count": improvement_count,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_ratio": wins / improvement_count,
                "loss_ratio": losses / improvement_count,
                "tie_ratio": ties / improvement_count,
                "avg_delta_ms": delta_sum / improvement_count,
            }

        return summary

    return {
        "iol_refresh": store.get("iol_refresh"),
        "yfinance": store.get("yfinance"),
        "fx_api": store.get("fx_api"),
        "fx_cache": store.get("fx_cache"),
        "portfolio": store.get("portfolio"),
        "quotes": store.get("quotes"),
        "opportunities": store.get("opportunities"),
        "opportunities_history": list(store.get(_OPPORTUNITIES_HISTORY_KEY, [])),
        "opportunities_stats": _summarize_stats(store.get(_OPPORTUNITIES_STATS_KEY)),
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
