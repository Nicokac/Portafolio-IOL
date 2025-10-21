"""Portfolio health metrics helpers."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Mapping, Optional

from .constants import _PORTFOLIO_HISTORY_LIMIT
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


def summarize_portfolio_stats(
    stats: Any,
    *,
    include_success: bool = True,
    include_latency: bool = True,
) -> Dict[str, Any]:
    """Return a consolidated summary of portfolio metrics."""

    if not isinstance(stats, Mapping):
        return {}

    summary: Dict[str, Any] = {}
    try:
        invocations = int(stats.get("invocations", 0) or 0)
    except (TypeError, ValueError):
        invocations = 0
    if invocations:
        summary["invocations"] = invocations

    if include_latency:
        latency_block = _summarize_metric_block(stats, "latency")
        if latency_block:
            summary["latency"] = latency_block

        try:
            latency_count = int(stats.get("latency_count", 0) or 0)
        except (TypeError, ValueError):
            latency_count = 0
        missing = invocations - latency_count
        if missing > 0:
            summary["missing_latency"] = missing

    if include_success:
        success_block = _summarize_metric_block(stats, "success")
        if success_block:
            summary["success"] = success_block

    sources = _normalize_counter_map(stats.get("sources"))
    if sources:
        total = sum(sources.values())
        source_data: Dict[str, Any] = {"counts": sources}
        if total:
            source_data["ratios"] = _compute_ratio_map(sources, total)
        summary["sources"] = source_data

    events = _serialize_event_history(stats.get("event_history"))
    if events:
        summary["events"] = events

    return summary


_summarize_portfolio_stats = summarize_portfolio_stats


def record_portfolio_load(
    elapsed_ms: Optional[float], *, source: str, detail: Optional[str] = None
) -> None:
    """Persist response time and source for the latest portfolio load."""
    store = get_store()
    source_text = str(source or "unknown").strip() or "unknown"
    detail_text = _clean_detail(detail)
    numeric_latency = _as_optional_float(elapsed_ms)
    now = time.time()

    summary: Dict[str, Any] = {
        "source": source_text,
        "elapsed_ms": float(numeric_latency) if numeric_latency is not None else None,
        "detail": detail_text,
        "ts": now,
    }

    stats_raw = store.get("portfolio_stats")
    stats: Dict[str, Any]
    if isinstance(stats_raw, Mapping):
        stats = dict(stats_raw)
    else:
        stats = {}

    stats["invocations"] = int(stats.get("invocations", 0) or 0) + 1

    source_counts_raw = stats.get("sources")
    if isinstance(source_counts_raw, Mapping):
        source_counts = dict(source_counts_raw)
    else:
        source_counts = {}
    source_counts[source_text] = int(source_counts.get(source_text, 0) or 0) + 1
    stats["sources"] = source_counts

    if numeric_latency is not None and math.isfinite(numeric_latency):
        value = float(numeric_latency)
        stats["latency_count"] = int(stats.get("latency_count", 0) or 0) + 1
        stats["latency_sum"] = float(stats.get("latency_sum", 0.0) or 0.0) + value
        stats["latency_sum_sq"] = (
            float(stats.get("latency_sum_sq", 0.0) or 0.0) + value * value
        )
        current_min = _as_optional_float(stats.get("latency_min"))
        stats["latency_min"] = value if current_min is None else min(current_min, value)
        current_max = _as_optional_float(stats.get("latency_max"))
        stats["latency_max"] = value if current_max is None else max(current_max, value)
        latency_history = _ensure_latency_history(
            stats.get("latency_history"), limit=_PORTFOLIO_HISTORY_LIMIT
        )
        latency_history.append(value)
        stats["latency_history"] = latency_history
        stats["last_elapsed_ms"] = value
    else:
        stats["missing_latency"] = int(stats.get("missing_latency", 0) or 0) + 1

    stats["last_source"] = source_text
    stats["last_detail"] = detail_text
    stats["last_ts"] = now

    latest_event = dict(summary)
    event_history = _ensure_event_history(
        stats.get("event_history"), limit=_PORTFOLIO_HISTORY_LIMIT
    )
    event_history.append(latest_event)
    stats["event_history"] = event_history

    store["portfolio_stats"] = stats

    metrics_summary = _summarize_portfolio_stats(stats)
    if metrics_summary:
        summary["stats"] = metrics_summary

    store["portfolio"] = summary

    log_analysis_event("portfolio.load", latest_event, metrics_summary)


def portfolio_metrics_snapshot(store: Mapping[str, Any]) -> Dict[str, Any]:
    """Return merged portfolio metrics for dashboards."""

    portfolio_data = _merge_entry(
        store.get("portfolio"), _summarize_portfolio_stats(store.get("portfolio_stats"))
    )
    return {"portfolio": portfolio_data}


__all__ = [
    "portfolio_metrics_snapshot",
    "record_portfolio_load",
    "summarize_portfolio_stats",
]
