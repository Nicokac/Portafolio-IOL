"""Helpers for FX-related health metrics."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Mapping, Optional

from shared.settings import cache_ttl_fx

from .constants import _FX_API_HISTORY_LIMIT, _FX_CACHE_HISTORY_LIMIT
from .session_adapter import get_store
from .telemetry import log_analysis_event
from .utils import (
    _as_optional_float,
    _clean_detail,
    _compute_ratio_map,
    _ensure_event_history,
    _ensure_latency_history,
    _merge_entry,
    _normalize_counter_map,
    _serialize_event_history,
    _summarize_metric_block,
)


def _classify_fx_cache_event(
    mode: str,
    age: Optional[float],
    stats: Mapping[str, Any],
) -> str:
    normalized_mode = str(mode or "unknown").strip().casefold() or "unknown"
    age_value = _as_optional_float(age)
    has_data = bool(stats.get("has_data"))

    if normalized_mode == "hit":
        if age_value is None:
            return "empty"
        if not math.isfinite(age_value):
            return "unknown"
        if age_value > cache_ttl_fx:
            return "stale"
        return "fresh"

    if normalized_mode == "refresh":
        return "stale" if has_data else "empty"

    return "unknown"


def _summarize_fx_api_stats(stats: Any) -> Dict[str, Any]:
    if not isinstance(stats, Mapping):
        return {}

    summary: Dict[str, Any] = {}

    try:
        invocations = int(stats.get("invocations", 0) or 0)
    except (TypeError, ValueError):
        invocations = 0
    if invocations:
        summary["invocations"] = invocations

    latency_block = _summarize_metric_block(stats, "latency")
    if latency_block:
        summary["latency"] = latency_block

    statuses = _normalize_counter_map(stats.get("status_counts"))
    if statuses:
        total = sum(statuses.values())
        status_payload: Dict[str, Any] = {"counts": statuses}
        if total:
            status_payload["ratios"] = _compute_ratio_map(statuses, total)
        summary["status"] = status_payload

    errors = _normalize_counter_map(stats.get("error_counts"))
    if errors:
        summary["errors"] = errors

    last_error = stats.get("last_error")
    if isinstance(last_error, str) and last_error:
        summary["last_error"] = last_error

    events = _serialize_event_history(stats.get("event_history"))
    if events:
        summary["events"] = events

    return summary


def _summarize_fx_cache_stats(stats: Any) -> Dict[str, Any]:
    if not isinstance(stats, Mapping):
        return {}

    summary: Dict[str, Any] = {}

    try:
        invocations = int(stats.get("invocations", 0) or 0)
    except (TypeError, ValueError):
        invocations = 0
    if invocations:
        summary["invocations"] = invocations

    modes = _normalize_counter_map(stats.get("mode_counts"))
    if modes:
        total_modes = sum(modes.values())
        mode_payload: Dict[str, Any] = {"counts": modes}
        if total_modes:
            mode_payload["ratios"] = _compute_ratio_map(modes, total_modes)
        summary["modes"] = mode_payload

    labels = _normalize_counter_map(stats.get("label_counts"))
    if labels:
        total_labels = sum(labels.values())
        label_payload: Dict[str, Any] = {"counts": labels}
        if total_labels:
            label_payload["ratios"] = _compute_ratio_map(labels, total_labels)
        summary["labels"] = label_payload

    age_block = _summarize_metric_block(stats, "age")
    if age_block:
        summary["age"] = age_block

    last_label = stats.get("last_label")
    if isinstance(last_label, str) and last_label:
        summary["last_label"] = last_label

    events = _serialize_event_history(stats.get("event_history"))
    if events:
        summary["events"] = events

    return summary


def record_fx_api_response(*, error: Optional[str] = None, elapsed_ms: Optional[float] = None) -> None:
    """Persist metadata about the latest FX API call."""

    store = get_store()
    status_text = "success" if not error else "error"
    error_text = _clean_detail(error)
    numeric_latency = _as_optional_float(elapsed_ms)
    now = time.time()

    summary: Dict[str, Any] = {
        "status": status_text,
        "error": error_text,
        "elapsed_ms": float(numeric_latency) if numeric_latency is not None else None,
        "ts": now,
    }

    stats_raw = store.get("fx_api_stats")
    stats: Dict[str, Any]
    if isinstance(stats_raw, Mapping):
        stats = dict(stats_raw)
    else:
        stats = {}

    stats["invocations"] = int(stats.get("invocations", 0) or 0) + 1

    status_counts_raw = stats.get("status_counts")
    if isinstance(status_counts_raw, Mapping):
        status_counts = dict(status_counts_raw)
    else:
        status_counts = {}
    status_counts[status_text] = int(status_counts.get(status_text, 0) or 0) + 1
    stats["status_counts"] = status_counts

    if error_text:
        error_counts_raw = stats.get("error_counts")
        if isinstance(error_counts_raw, Mapping):
            error_counts = dict(error_counts_raw)
        else:
            error_counts = {}
        error_counts[error_text] = int(error_counts.get(error_text, 0) or 0) + 1
        stats["error_counts"] = error_counts
        stats["last_error"] = error_text

    if numeric_latency is not None and math.isfinite(numeric_latency):
        value = float(numeric_latency)
        stats["latency_count"] = int(stats.get("latency_count", 0) or 0) + 1
        stats["latency_sum"] = float(stats.get("latency_sum", 0.0) or 0.0) + value
        stats["latency_sum_sq"] = float(stats.get("latency_sum_sq", 0.0) or 0.0) + value * value
        current_min = _as_optional_float(stats.get("latency_min"))
        stats["latency_min"] = value if current_min is None else min(current_min, value)
        current_max = _as_optional_float(stats.get("latency_max"))
        stats["latency_max"] = value if current_max is None else max(current_max, value)
        latency_history = _ensure_latency_history(stats.get("latency_history"), limit=_FX_API_HISTORY_LIMIT)
        latency_history.append(value)
        stats["latency_history"] = latency_history
        stats["last_elapsed_ms"] = value
    else:
        stats["missing_latency"] = int(stats.get("missing_latency", 0) or 0) + 1

    stats["last_status"] = status_text
    stats["last_ts"] = now

    latest_event = dict(summary)
    event_history = _ensure_event_history(stats.get("event_history"), limit=_FX_API_HISTORY_LIMIT)
    event_history.append(latest_event)
    stats["event_history"] = event_history

    store["fx_api_stats"] = stats

    metrics_summary = _summarize_fx_api_stats(stats)
    if metrics_summary:
        summary["stats"] = metrics_summary

    store["fx_api"] = summary

    log_analysis_event("fx.api", latest_event, metrics_summary)


def record_fx_cache_usage(mode: str, *, age: Optional[float] = None) -> None:
    """Persist information about session cache usage for FX rates."""
    store = get_store()

    mode_text = str(mode or "unknown").strip() or "unknown"
    mode_key = mode_text.casefold()
    numeric_age = _as_optional_float(age)
    now = time.time()

    entry: Dict[str, Any] = {
        "mode": mode_text,
        "age": float(numeric_age) if numeric_age is not None else None,
        "ts": now,
    }

    stats_raw = store.get("fx_cache_stats")
    stats: Dict[str, Any]
    if isinstance(stats_raw, Mapping):
        stats = dict(stats_raw)
    else:
        stats = {}

    stats["invocations"] = int(stats.get("invocations", 0) or 0) + 1

    mode_counts_raw = stats.get("mode_counts")
    if isinstance(mode_counts_raw, Mapping):
        mode_counts = dict(mode_counts_raw)
    else:
        mode_counts = {}
    mode_counts[mode_key] = int(mode_counts.get(mode_key, 0) or 0) + 1
    stats["mode_counts"] = mode_counts

    classification = _classify_fx_cache_event(mode_text, numeric_age, stats)
    if classification:
        entry["label"] = classification
        label_counts_raw = stats.get("label_counts")
        if isinstance(label_counts_raw, Mapping):
            label_counts = dict(label_counts_raw)
        else:
            label_counts = {}
        label_counts[classification] = int(label_counts.get(classification, 0) or 0) + 1
        stats["label_counts"] = label_counts
        stats["last_label"] = classification

    if mode_key in {"hit", "refresh"}:
        stats["has_data"] = True

    stats["last_mode"] = mode_key
    if numeric_age is not None and math.isfinite(numeric_age):
        stats["age_count"] = int(stats.get("age_count", 0) or 0) + 1
        stats["age_sum"] = float(stats.get("age_sum", 0.0) or 0.0) + numeric_age
        stats["age_sum_sq"] = float(stats.get("age_sum_sq", 0.0) or 0.0) + numeric_age * numeric_age
        current_min = _as_optional_float(stats.get("age_min"))
        stats["age_min"] = numeric_age if current_min is None else min(current_min, numeric_age)
        current_max = _as_optional_float(stats.get("age_max"))
        stats["age_max"] = numeric_age if current_max is None else max(current_max, numeric_age)
        age_history = _ensure_latency_history(stats.get("age_history"), limit=_FX_CACHE_HISTORY_LIMIT)
        age_history.append(numeric_age)
        stats["age_history"] = age_history

    stats["last_age"] = numeric_age

    event_history = _ensure_event_history(stats.get("event_history"), limit=_FX_CACHE_HISTORY_LIMIT)
    latest_event = dict(entry)
    event_history.append(latest_event)
    stats["event_history"] = event_history

    store["fx_cache_stats"] = stats

    summary = _summarize_fx_cache_stats(stats)
    if summary:
        entry["stats"] = summary

    store["fx_cache"] = entry

    log_analysis_event("fx.cache", latest_event, summary)


def fx_metrics_snapshot(store: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the merged FX metrics ready for health dashboards."""

    fx_api_data = _merge_entry(store.get("fx_api"), _summarize_fx_api_stats(store.get("fx_api_stats")))
    fx_cache_data = _merge_entry(store.get("fx_cache"), _summarize_fx_cache_stats(store.get("fx_cache_stats")))

    return {
        "fx_api": fx_api_data,
        "fx_cache": fx_cache_data,
    }


__all__ = [
    "fx_metrics_snapshot",
    "record_fx_api_response",
    "record_fx_cache_usage",
]
