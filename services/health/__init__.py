from __future__ import annotations

"""Helpers to capture health metrics and expose them via ``st.session_state``."""

from collections import deque
import logging
import math
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Sequence
import time

from .constants import (
    _ACTIVE_SESSIONS_KEY,
    _ADAPTER_FALLBACK_KEY,
    _DEPENDENCIES_KEY,
    _DIAGNOSTICS_SNAPSHOT_KEY,
    _ENVIRONMENT_SNAPSHOT_KEY,
    _FX_API_HISTORY_LIMIT,
    _FX_CACHE_HISTORY_LIMIT,
    _HEALTH_KEY,
    _LAST_HTTP_ERROR_KEY,
    _LATENCY_FAST_THRESHOLD_MS,
    _LATENCY_MEDIUM_THRESHOLD_MS,
    _LOGIN_TO_RENDER_STATS_KEY,
    _MARKET_DATA_INCIDENT_LIMIT,
    _MARKET_DATA_INCIDENTS_KEY,
    _PORTFOLIO_HISTORY_LIMIT,
    _PROVIDER_HISTORY_LIMIT,
    _PROVIDER_LABELS,
    _QUOTE_HISTORY_LIMIT,
    _QUOTE_PROVIDER_HISTORY_LIMIT,
    _QUOTE_RATE_LIMIT_KEY,
    _RISK_INCIDENTS_KEY,
    _RISK_INCIDENT_HISTORY_LIMIT,
    _SESSION_MONITORING_KEY,
    _SESSION_MONITORING_TTL_SECONDS,
    _SNAPSHOT_EVENT_KEY,
    _TAB_LATENCIES_KEY,
    _TAB_LATENCY_HISTORY_LIMIT,
)
from .metrics_fx import (
    fx_metrics_snapshot,
    record_fx_api_response,
    record_fx_cache_usage,
)
from .metrics_portfolio import (
    portfolio_metrics_snapshot,
    record_portfolio_load,
)
from .session_adapter import (
    ensure_session_monitoring_store as _ensure_session_monitoring_store,
    get_store as _store,
    record_http_error,
    record_login_to_render,
    record_session_started,
    st,
)
from .snapshots import (
    record_environment_snapshot,
    record_snapshot_event,
    snapshot_event_summary,
)
from .telemetry import log_analysis_event as _log_analysis_event
from .utils import (
    _as_optional_float,
    _as_optional_int,
    _clean_detail,
    _compute_ratio_map,
    _ensure_event_history,
    _ensure_history_deque,
    _ensure_latency_history,
    _ensure_sequence,
    _merge_entry,
    _normalize_backend_details,
    _normalize_counter_map,
    _normalize_environment_snapshot,
    _normalize_metadata,
    _serialize_event_history,
    _summarize_metric_block,
)


logger = logging.getLogger(__name__)
analysis_logger = logging.getLogger("analysis")


 


def _normalize_diagnostics_snapshot_entry(
    result: Mapping[str, Any],
    *,
    source: Optional[str] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    if now is None:
        now = time.time()

    status_text = str(result.get("status") or "unknown").strip() or "unknown"
    latency_value = _as_optional_float(result.get("latency"))
    ts_value = _as_optional_float(result.get("timestamp"))
    if ts_value is None:
        ts_value = _as_optional_float(result.get("ts"))
    if ts_value is None:
        ts_value = now

    normalized_snapshot: Dict[str, Any] = {"status": status_text}
    entry: Dict[str, Any] = {"status": status_text, "ts": ts_value}
    if latency_value is not None:
        entry["latency"] = latency_value
        normalized_snapshot["latency"] = latency_value

    component = result.get("component")
    if component is not None:
        component_text = str(component).strip()
        if component_text:
            entry["component"] = component_text
            normalized_snapshot["component"] = component_text

    message_value = result.get("message")
    if message_value is not None:
        message_text = _clean_detail(message_value)
        if message_text:
            entry["message"] = message_text
            normalized_snapshot["message"] = message_text

    checks_raw = result.get("checks")
    if isinstance(checks_raw, Iterable) and not isinstance(
        checks_raw, (str, bytes, bytearray)
    ):
        checks: list[Dict[str, Any]] = []
        for item in checks_raw:
            if not isinstance(item, Mapping):
                continue
            check_entry: Dict[str, Any] = {}
            component_value = item.get("component") or item.get("name")
            if component_value is not None:
                component_text = str(component_value).strip()
                if component_text:
                    check_entry["component"] = component_text
            status_value = item.get("status")
            if status_value is not None:
                status_text = str(status_value).strip()
                if status_text:
                    check_entry["status"] = status_text
            message_data = item.get("message")
            detail_text = _clean_detail(message_data)
            if detail_text:
                check_entry["message"] = detail_text
            if check_entry:
                checks.append(check_entry)
        if checks:
            entry["checks"] = checks
            normalized_snapshot["checks"] = checks

    source_text = str(source or "").strip()
    if source_text:
        entry["source"] = source_text

    entry["snapshot"] = normalized_snapshot
    return entry


def record_diagnostics_snapshot(
    result: Mapping[str, Any],
    *,
    source: Optional[str] = None,
) -> None:
    """Persist the latest startup diagnostics summary."""

    if not isinstance(result, Mapping):
        result = {}

    store = _store()
    now = time.time()

    try:
        entry = _normalize_diagnostics_snapshot_entry(result, source=source, now=now)
        store[_DIAGNOSTICS_SNAPSHOT_KEY] = entry

        normalized_snapshot = entry.get("snapshot")
        metrics: Dict[str, Any] = {}
        if isinstance(normalized_snapshot, dict):
            metrics["field_count"] = len(normalized_snapshot)

        if metrics:
            _log_analysis_event("diagnostics_snapshot", entry, metrics)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "⚠️ No se pudo registrar el diagnóstico de inicio",
            exc_info=exc,
        )
        fallback_entry: Dict[str, Any] = {
            "status": "unknown",
            "ts": now,
            "snapshot": {"status": "unknown"},
        }
        source_text = str(source or "").strip()
        if source_text:
            fallback_entry["source"] = source_text
        store[_DIAGNOSTICS_SNAPSHOT_KEY] = fallback_entry


def record_dependency_status(
    name: str,
    *,
    status: str,
    detail: Optional[str] = None,
    label: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    """Persist the latest status for a critical runtime dependency."""

    dependency_key = str(name or "").strip() or "unknown"
    status_text = str(status or "unknown").strip() or "unknown"
    label_text = str(label or name or "").strip() or dependency_key
    detail_text = _clean_detail(detail)
    source_text = str(source or "").strip()

    store = _store()
    dependencies = store.setdefault(_DEPENDENCIES_KEY, {})

    entry: Dict[str, Any] = {
        "status": status_text,
        "label": label_text,
        "ts": time.time(),
    }
    if detail_text:
        entry["detail"] = detail_text
    if source_text:
        entry["source"] = source_text

    dependencies[dependency_key] = entry


def record_iol_refresh(success: bool, *, detail: Optional[str] = None) -> None:
    """Persist the outcome of the last IOL login/refresh attempt."""
    store = _store()
    store["iol_refresh"] = {
        "status": "success" if success else "error",
        "detail": _clean_detail(detail),
        "ts": time.time(),
    }


def record_quote_rate_limit_wait(provider: str, wait_seconds: float, *, reason: str) -> None:
    """Track waiting time introduced by quote provider rate limiting."""

    provider_label = str(provider or "unknown").strip() or "unknown"
    provider_key = provider_label.casefold()
    wait = _as_optional_float(wait_seconds)
    if wait is None:
        return

    store = _store()
    raw_limits = store.get(_QUOTE_RATE_LIMIT_KEY)
    if isinstance(raw_limits, Mapping):
        limits = dict(raw_limits)
    else:
        limits = {}

    raw_entry = limits.get(provider_key)
    if isinstance(raw_entry, Mapping):
        entry = dict(raw_entry)
    else:
        entry = {}

    entry["provider"] = provider_key
    entry["label"] = _PROVIDER_LABELS.get(provider_key, provider_label)
    entry["count"] = int(entry.get("count", 0) or 0) + 1
    entry["wait_total"] = float(entry.get("wait_total", 0.0) or 0.0) + wait
    entry["wait_last"] = wait
    entry["last_reason"] = str(reason or "throttle").strip() or "throttle"
    entry["ts"] = time.time()

    reason_counts = entry.get("reason_counts")
    if isinstance(reason_counts, Mapping):
        reason_counts = dict(reason_counts)
    else:
        reason_counts = {}
    reason_key = entry["last_reason"]
    reason_counts[reason_key] = int(reason_counts.get(reason_key, 0) or 0) + 1
    entry["reason_counts"] = reason_counts

    limits[provider_key] = entry
    store[_QUOTE_RATE_LIMIT_KEY] = limits


def _ensure_history_deque(raw_history: Any, *, limit: int) -> Deque[Dict[str, Any]]:
    """Return a deque that can be safely reused to store provider history."""

    if isinstance(raw_history, deque) and raw_history.maxlen == limit:
        return raw_history

    history: Deque[Dict[str, Any]] = deque(maxlen=limit)
    if isinstance(raw_history, Iterable) and not isinstance(
        raw_history, (str, bytes, bytearray)
    ):
        for entry in raw_history:
            normalized = _normalize_provider_event(entry)
            if normalized is not None:
                history.append(normalized)
    return history


def _ensure_latency_history(raw_history: Any, *, limit: int) -> Deque[float]:
    """Return a deque storing float latencies with bounded size."""

    if isinstance(raw_history, deque) and raw_history.maxlen == limit:
        return raw_history

    history: Deque[float] = deque(maxlen=limit)
    if isinstance(raw_history, Iterable) and not isinstance(
        raw_history, (str, bytes, bytearray)
    ):
        for value in raw_history:
            numeric = _as_optional_float(value)
            if numeric is None:
                continue
            history.append(float(numeric))
    return history


def _ensure_event_history(raw_history: Any, *, limit: int) -> Deque[Dict[str, Any]]:
    """Return a deque that keeps track of the latest metric events."""

    if isinstance(raw_history, deque) and raw_history.maxlen == limit:
        return raw_history

    history: Deque[Dict[str, Any]] = deque(maxlen=limit)
    if isinstance(raw_history, Iterable) and not isinstance(
        raw_history, (str, bytes, bytearray)
    ):
        for entry in raw_history:
            if isinstance(entry, Mapping):
                history.append(dict(entry))
    return history


def _normalize_numeric(value: float) -> float | int:
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _serialize_event_history(raw_history: Any) -> list[Dict[str, Any]]:
    if isinstance(raw_history, deque):
        iterable = raw_history
    elif isinstance(raw_history, Iterable) and not isinstance(
        raw_history, (str, bytes, bytearray)
    ):
        iterable = raw_history
    else:
        return []

    serialized: list[Dict[str, Any]] = []
    for entry in iterable:
        if isinstance(entry, Mapping):
            serialized.append(dict(entry))
    return serialized


def _summarize_metric_block(
    stats: Mapping[str, Any],
    prefix: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(stats, Mapping):
        return None

    try:
        count = int(stats.get(f"{prefix}_count", 0) or 0)
    except (TypeError, ValueError):
        count = 0
    if count <= 0:
        return None

    try:
        sum_value = float(stats.get(f"{prefix}_sum", 0.0) or 0.0)
    except (TypeError, ValueError):
        sum_value = 0.0

    try:
        sum_sq_value = float(stats.get(f"{prefix}_sum_sq", 0.0) or 0.0)
    except (TypeError, ValueError):
        sum_sq_value = 0.0

    avg = sum_value / count
    variance = max(sum_sq_value / count - avg * avg, 0.0)
    block: Dict[str, Any] = {
        "count": count,
        "avg": avg,
        "stdev": math.sqrt(variance),
    }

    min_value = _as_optional_float(stats.get(f"{prefix}_min"))
    if min_value is not None and math.isfinite(min_value):
        block["min"] = float(min_value)
    max_value = _as_optional_float(stats.get(f"{prefix}_max"))
    if max_value is not None and math.isfinite(max_value):
        block["max"] = float(max_value)

    history_key = f"{prefix}_history"
    history_raw = stats.get(history_key)
    if isinstance(history_raw, deque):
        samples: list[float | int] = []
        for value in history_raw:
            numeric = _as_optional_float(value)
            if numeric is None or not math.isfinite(numeric):
                continue
            samples.append(_normalize_numeric(numeric))
        if samples:
            block["samples"] = samples

    return block


def _normalize_counter_map(raw_map: Any) -> Dict[str, int]:
    if not isinstance(raw_map, Mapping):
        return {}
    counters: Dict[str, int] = {}
    for key, value in raw_map.items():
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            continue
        if numeric < 0:
            continue
        name = str(key).strip()
        if not name:
            continue
        counters[name] = numeric
    return counters


def _summarize_portfolio_stats(stats: Any) -> Dict[str, Any]:
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

    latency_count = int(stats.get("latency_count", 0) or 0)
    missing = invocations - latency_count
    if missing > 0:
        summary["missing_latency"] = missing

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


def _summarize_quote_stats(stats: Any) -> Dict[str, Any]:
    if not isinstance(stats, Mapping):
        return {}

    summary = _summarize_portfolio_stats(stats)

    batch_block = _summarize_metric_block(stats, "batch")
    if batch_block:
        summary["batch"] = batch_block

    return summary


def record_yfinance_usage(
    source: str,
    *,
    detail: Optional[str] = None,
    status: Optional[str] = None,
    fallback: Optional[bool] = None,
) -> None:
    """Persist whether Yahoo Finance or a fallback served the last request."""

    provider = str(source or "unknown").strip() or "unknown"
    now = time.time()
    detail_text = _clean_detail(detail)
    fallback_flag = bool(fallback) if fallback is not None else provider.casefold() != "yfinance"
    status_text_raw = str(status or "").strip().casefold()
    if not status_text_raw:
        status_text_raw = "fallback" if fallback_flag else "success"

    store = _store()
    existing = store.get("yfinance")
    data: Dict[str, Any]
    if isinstance(existing, Mapping):
        data = dict(existing)
    else:
        data = {}

    history = _ensure_history_deque(data.get("history"), limit=_PROVIDER_HISTORY_LIMIT)
    event: Dict[str, Any] = {
        "provider": provider,
        "result": status_text_raw,
        "fallback": fallback_flag,
        "ts": now,
    }
    if detail_text:
        event["detail"] = detail_text
    history.append(event)

    data["source"] = provider
    data["ts"] = now
    data["history"] = history
    data["fallback"] = fallback_flag
    data["latest_provider"] = provider
    data["latest_result"] = status_text_raw
    data["result"] = status_text_raw
    if detail_text is not None:
        data["detail"] = detail_text
    elif "detail" in data:
        del data["detail"]

    store["yfinance"] = data


def record_market_data_incident(
    *,
    adapter: str,
    provider: str,
    status: str,
    detail: Optional[str] = None,
    fallback: bool | None = None,
) -> None:
    """Persist incidents for market data adapters."""

    entry = {
        "adapter": str(adapter),
        "provider": str(provider),
        "status": str(status),
        "ts": time.time(),
    }
    detail_text = _clean_detail(detail)
    if detail_text:
        entry["detail"] = detail_text
    if fallback is not None:
        entry["fallback"] = bool(fallback)

    store = _store()
    incidents = list(store.get(_MARKET_DATA_INCIDENTS_KEY, []))
    incidents.append(entry)
    store[_MARKET_DATA_INCIDENTS_KEY] = incidents[-_MARKET_DATA_INCIDENT_LIMIT:]


def record_risk_incident(
    *,
    category: str,
    severity: str = "warning",
    detail: Optional[str] = None,
    fallback: bool | None = None,
    source: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Persist structured information about detected risk incidents."""

    now = time.time()
    category_text = str(category or "unknown").strip() or "unknown"
    severity_raw = str(severity or "unknown").strip() or "unknown"
    severity_key = severity_raw.casefold() or "unknown"
    detail_text = _clean_detail(detail)
    source_text = _clean_detail(source)

    if tags is not None:
        tag_list = []
        for tag in tags:
            text = str(tag).strip()
            if text:
                tag_list.append(text)
        if not tag_list:
            tag_list = None
    else:
        tag_list = None

    metadata_payload: Optional[Dict[str, Any]] = None
    if isinstance(metadata, Mapping):
        metadata_payload = {}
        for key, value in metadata.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            metadata_payload[key_text] = value
        if not metadata_payload:
            metadata_payload = None

    fallback_flag: Optional[bool]
    if fallback is None:
        fallback_flag = None
    else:
        fallback_flag = bool(fallback)

    entry: Dict[str, Any] = {
        "category": category_text,
        "severity": severity_key,
        "severity_label": severity_raw,
        "ts": now,
    }
    if detail_text:
        entry["detail"] = detail_text
    if fallback_flag is not None:
        entry["fallback"] = fallback_flag
    if source_text:
        entry["source"] = source_text
    if tag_list:
        entry["tags"] = tag_list
    if metadata_payload is not None:
        entry["metadata"] = metadata_payload

    store = _store()
    raw_risk = store.get(_RISK_INCIDENTS_KEY)
    if isinstance(raw_risk, Mapping):
        risk_data = dict(raw_risk)
    else:
        risk_data = {}

    total = int(risk_data.get("total", 0) or 0) + 1
    risk_data["total"] = total

    fallback_total = int(risk_data.get("fallback_count", 0) or 0)
    if fallback_flag:
        fallback_total += 1
    risk_data["fallback_count"] = fallback_total
    risk_data["fallback_ratio"] = (fallback_total / total) if total else 0.0

    raw_categories = risk_data.get("by_category")
    if isinstance(raw_categories, Mapping):
        categories: Dict[str, Any] = {
            str(key): dict(value)
            for key, value in raw_categories.items()
            if isinstance(value, Mapping)
        }
    else:
        categories = {}

    category_stats = categories.get(category_text, {})
    if not isinstance(category_stats, Mapping):
        category_stats = {}
    else:
        category_stats = dict(category_stats)

    category_total = int(category_stats.get("count", 0) or 0) + 1
    category_stats["label"] = category_stats.get("label") or category_text
    category_stats["count"] = category_total

    category_fallback = int(category_stats.get("fallback_count", 0) or 0)
    if fallback_flag:
        category_fallback += 1
    category_stats["fallback_count"] = category_fallback
    category_stats["fallback_ratio"] = (
        category_fallback / category_total if category_total else 0.0
    )

    severity_counts_raw = category_stats.get("severity_counts")
    if isinstance(severity_counts_raw, Mapping):
        severity_counts = {
            str(key): int(value)
            for key, value in severity_counts_raw.items()
            if _as_optional_int(value) is not None
        }
    else:
        severity_counts = {}
    severity_counts[severity_key] = int(severity_counts.get(severity_key, 0)) + 1
    category_stats["severity_counts"] = severity_counts
    category_stats["severity_ratios"] = _compute_ratio_map(
        severity_counts, category_total
    )

    category_stats["last_severity"] = severity_key
    category_stats["last_ts"] = now
    if detail_text:
        category_stats["last_detail"] = detail_text
    if fallback_flag is not None:
        category_stats["last_fallback"] = fallback_flag
    if source_text:
        category_stats["last_source"] = source_text
    if tag_list:
        category_stats["last_tags"] = tag_list

    categories[category_text] = category_stats
    risk_data["by_category"] = categories

    raw_severities = risk_data.get("by_severity")
    if isinstance(raw_severities, Mapping):
        severities: Dict[str, Any] = {
            str(key): dict(value)
            for key, value in raw_severities.items()
            if isinstance(value, Mapping)
        }
    else:
        severities = {}

    severity_stats = severities.get(severity_key, {})
    if not isinstance(severity_stats, Mapping):
        severity_stats = {}
    else:
        severity_stats = dict(severity_stats)

    severity_total = int(severity_stats.get("count", 0) or 0) + 1
    severity_stats["label"] = severity_stats.get("label") or severity_raw
    severity_stats["count"] = severity_total

    severity_fallback = int(severity_stats.get("fallback_count", 0) or 0)
    if fallback_flag:
        severity_fallback += 1
    severity_stats["fallback_count"] = severity_fallback
    severity_stats["fallback_ratio"] = (
        severity_fallback / severity_total if severity_total else 0.0
    )

    categories_by_severity_raw = severity_stats.get("categories")
    if isinstance(categories_by_severity_raw, Mapping):
        categories_by_severity = {
            str(key): int(value)
            for key, value in categories_by_severity_raw.items()
            if _as_optional_int(value) is not None
        }
    else:
        categories_by_severity = {}
    categories_by_severity[category_text] = (
        int(categories_by_severity.get(category_text, 0)) + 1
    )
    severity_stats["categories"] = categories_by_severity
    severity_stats["category_ratios"] = _compute_ratio_map(
        categories_by_severity, severity_total
    )
    severity_stats["last_ts"] = now
    severity_stats["last_category"] = category_text

    severities[severity_key] = severity_stats
    risk_data["by_severity"] = severities

    latest_entry = dict(entry)
    risk_data["latest"] = latest_entry

    raw_latest_by_category = risk_data.get("latest_by_category")
    if isinstance(raw_latest_by_category, Mapping):
        latest_by_category: Dict[str, Any] = {
            str(key): dict(value)
            for key, value in raw_latest_by_category.items()
            if isinstance(value, Mapping)
        }
    else:
        latest_by_category = {}
    latest_by_category[category_text] = latest_entry
    risk_data["latest_by_category"] = latest_by_category

    history_raw = risk_data.get("history")
    if isinstance(history_raw, deque):
        history = history_raw
    elif isinstance(history_raw, Iterable) and not isinstance(
        history_raw, (str, bytes, bytearray)
    ):
        history = deque(history_raw, maxlen=_RISK_INCIDENT_HISTORY_LIMIT)
    else:
        history = deque(maxlen=_RISK_INCIDENT_HISTORY_LIMIT)
    history.append(latest_entry)
    risk_data["history"] = history

    store[_RISK_INCIDENTS_KEY] = risk_data


def record_macro_api_usage(
    *,
    attempts: Iterable[Mapping[str, Any]],
    notes: Optional[Iterable[str]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    latest: Optional[Mapping[str, Any]] = None,
) -> None:
    """Persist information about the macro/sector data providers."""

    now = time.time()

    def _normalize_attempt(entry: Mapping[str, Any]) -> Dict[str, Any]:
        provider_raw = entry.get("provider")
        provider_text = str(provider_raw or "unknown").strip() or "unknown"
        provider_key = provider_text.casefold()

        label_raw = entry.get("label") or entry.get("provider_label")
        label = str(label_raw or provider_text).strip() or provider_text

        status_raw = str(entry.get("status") or "unknown").strip() or "unknown"
        status_value = status_raw.casefold() or "unknown"

        normalized: Dict[str, Any] = {
            "provider": provider_key,
            "provider_label": label,
            "status": status_raw,
            "provider_key": provider_key,
            "status_normalized": status_value,
            "provider_display": provider_text,
        }

        elapsed_value = _as_optional_float(entry.get("elapsed_ms"))
        if elapsed_value is not None:
            normalized["elapsed_ms"] = elapsed_value

        detail = _clean_detail(entry.get("detail"))
        if detail:
            normalized["detail"] = detail

        if entry.get("fallback"):
            normalized["fallback"] = True

        missing = _normalize_sectors(entry.get("missing_series"))
        if missing:
            normalized["missing_series"] = missing

        timestamp = entry.get("ts")
        try:
            normalized["ts"] = float(timestamp)
        except (TypeError, ValueError):
            normalized["ts"] = now

        return normalized

    normalized_attempts: list[Dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            continue
        normalized_attempts.append(_normalize_attempt(attempt))

    normalized_latest: Optional[Dict[str, Any]] = None
    if isinstance(latest, Mapping):
        normalized_latest = _normalize_attempt(latest)
    elif normalized_attempts:
        normalized_latest = dict(normalized_attempts[-1])

    normalized_notes: Optional[list[str]] = None
    if notes is not None:
        normalized_notes = [str(note).strip() for note in notes if str(note).strip()]

    normalized_metrics: Optional[Dict[str, Any]] = None
    if metrics is not None:
        normalized_metrics = dict(metrics)

    store = _store()
    raw_macro = store.get("macro_api")
    macro_data: Dict[str, Any]
    if isinstance(raw_macro, Mapping):
        macro_data = dict(raw_macro)
    else:
        macro_data = {}

    macro_data["ts"] = now
    macro_data["attempts"] = normalized_attempts
    if normalized_notes is not None:
        macro_data["notes"] = normalized_notes
    if normalized_metrics is not None:
        macro_data["metrics"] = normalized_metrics
    if normalized_latest is not None:
        macro_data["latest"] = normalized_latest

    raw_providers = macro_data.get("providers")
    providers: Dict[str, Any] = {}
    if isinstance(raw_providers, Mapping):
        for key, value in raw_providers.items():
            if isinstance(value, Mapping):
                providers[key] = dict(value)

    def _update_provider_stats(entry: Mapping[str, Any]) -> None:
        provider_key = str(
            entry.get("provider_key") or entry.get("provider") or "unknown"
        ).casefold()
        provider_label = str(
            entry.get("provider_label") or entry.get("label") or provider_key
        ).strip() or provider_key
        status_value = str(
            entry.get("status_normalized") or entry.get("status") or "unknown"
        ).casefold() or "unknown"
        elapsed_value = _as_optional_float(entry.get("elapsed_ms"))
        fallback_flag = bool(entry.get("fallback"))
        detail_value = _clean_detail(entry.get("detail"))
        missing_series = _normalize_sectors(entry.get("missing_series"))

        provider_stats_raw = providers.get(provider_key)
        if isinstance(provider_stats_raw, Mapping):
            provider_stats = dict(provider_stats_raw)
        else:
            provider_stats = {}

        latest_payload: Dict[str, Any] = {
            "provider": provider_key,
            "provider_label": provider_label,
            "status": status_value,
            "fallback": fallback_flag,
        }
        if elapsed_value is not None:
            latest_payload["elapsed_ms"] = elapsed_value
        ts_value = entry.get("ts")
        try:
            latest_payload["ts"] = float(ts_value)
        except (TypeError, ValueError):
            latest_payload["ts"] = now
        if detail_value:
            latest_payload["detail"] = detail_value
        if missing_series:
            latest_payload["missing_series"] = missing_series

        provider_stats["latest"] = latest_payload
        provider_stats["label"] = provider_label
        provider_stats["total"] = int(provider_stats.get("total", 0) or 0) + 1

        status_counts = provider_stats.get("status_counts")
        if not isinstance(status_counts, dict):
            status_counts = {}
        status_counts[status_value] = int(status_counts.get(status_value, 0) or 0) + 1
        provider_stats["status_counts"] = status_counts

        _increment_latency_bucket(provider_stats, "latency", elapsed_value)

        if fallback_flag:
            provider_stats["fallback_count"] = int(
                provider_stats.get("fallback_count", 0) or 0
            ) + 1

        if status_value == "error":
            provider_stats["error_count"] = int(
                provider_stats.get("error_count", 0) or 0
            ) + 1

        history_raw = provider_stats.get("history")
        if isinstance(history_raw, deque):
            history = deque(history_raw, maxlen=_PROVIDER_HISTORY_LIMIT)
        elif isinstance(history_raw, Iterable) and not isinstance(
            history_raw, (bytes, bytearray, str)
        ):
            history = deque(history_raw, maxlen=_PROVIDER_HISTORY_LIMIT)
        else:
            history = deque(maxlen=_PROVIDER_HISTORY_LIMIT)
        if status_value != "success" or fallback_flag:
            history.append(
                {
                    "status": status_value,
                    "detail": detail_value,
                    "fallback": fallback_flag,
                    "elapsed_ms": elapsed_value,
                    "missing_series": missing_series,
                    "ts": latest_payload.get("ts", now),
                }
            )
        provider_stats["history"] = list(history)

        total = int(provider_stats.get("total", 0) or 0)
        provider_stats["count"] = total
        provider_stats["status_counts"] = {
            str(key): int(value) for key, value in status_counts.items()
        }
        provider_stats["status_ratios"] = _compute_ratio_map(status_counts, total)

        error_total = int(provider_stats.get("error_count", 0) or 0)
        provider_stats["error_count"] = error_total
        provider_stats["error_ratio"] = (error_total / total) if total else 0.0

        fallback_total = int(provider_stats.get("fallback_count", 0) or 0)
        provider_stats["fallback_count"] = fallback_total
        provider_stats["fallback_ratio"] = (fallback_total / total) if total else 0.0

        latency_raw = provider_stats.get("latency_buckets")
        if isinstance(latency_raw, Mapping):
            latency_data = dict(latency_raw)
        else:
            latency_data = {}
        counts_raw = latency_data.get("counts")
        if isinstance(counts_raw, Mapping):
            counts = {str(key): int(counts_raw.get(key, 0) or 0) for key in counts_raw}
        else:
            counts = {}
        latency_data["counts"] = counts
        latency_data["total"] = total
        latency_data["ratios"] = _compute_ratio_map(counts, total)
        provider_stats["latency_buckets"] = latency_data

        providers[provider_key] = provider_stats

    for entry in normalized_attempts:
        _update_provider_stats(entry)

    macro_data["providers"] = providers
    overall = _aggregate_provider_overall(providers)
    if overall:
        macro_data["overall"] = overall
    store["macro_api"] = macro_data


def record_tab_latency(
    tab: str,
    elapsed_ms: Optional[float],
    *,
    status: str = "success",
) -> None:
    """Track latency distributions per analytical tab."""

    tab_key_raw = str(tab or "").strip()
    tab_key = tab_key_raw.casefold() or "unknown"
    status_key = str(status or "").strip().casefold() or "unknown"

    store = _store()
    raw_tabs = store.get(_TAB_LATENCIES_KEY)
    if isinstance(raw_tabs, Mapping):
        tabs = dict(raw_tabs)
    else:
        tabs = {}

    raw_stats = tabs.get(tab_key)
    if isinstance(raw_stats, Mapping):
        stats = dict(raw_stats)
    else:
        stats = {}

    stats["label"] = tab_key_raw or tab_key.title()

    total_invocations = int(stats.get("total", 0) or 0) + 1
    stats["total"] = total_invocations

    numeric_latency = _as_optional_float(elapsed_ms)
    if numeric_latency is not None and math.isfinite(numeric_latency):
        value = float(numeric_latency)
        history = _ensure_latency_history(
            stats.get("history"), limit=_TAB_LATENCY_HISTORY_LIMIT
        )
        history.append(value)
        stats["history"] = history
        stats["count"] = int(stats.get("count", 0) or 0) + 1
        stats["sum"] = float(stats.get("sum", 0.0) or 0.0) + value
        stats["sum_sq"] = float(stats.get("sum_sq", 0.0) or 0.0) + value * value
        stats["last_elapsed_ms"] = value
    else:
        stats["missing_count"] = int(stats.get("missing_count", 0) or 0) + 1
        stats["last_elapsed_ms"] = None

    status_counts = stats.get("status_counts")
    if isinstance(status_counts, Mapping):
        status_counts = dict(status_counts)
    else:
        status_counts = {}
    status_counts[status_key] = int(status_counts.get(status_key, 0) or 0) + 1
    stats["status_counts"] = status_counts

    if status_key == "error":
        stats["error_count"] = int(stats.get("error_count", 0) or 0) + 1

    stats["last_status"] = status_key
    stats["ts"] = time.time()

    tabs[tab_key] = stats
    store[_TAB_LATENCIES_KEY] = tabs


def record_adapter_fallback(
    adapter: str,
    provider: str,
    status: str,
    fallback: bool,
) -> None:
    """Aggregate fallback ratios per adapter/provider combination."""

    adapter_label = str(adapter or "").strip() or "desconocido"
    adapter_key = adapter_label.casefold()
    provider_label = str(provider or "").strip() or "desconocido"
    provider_key = provider_label.casefold()
    status_key = str(status or "").strip().casefold() or "unknown"

    store = _store()
    raw_adapters = store.get(_ADAPTER_FALLBACK_KEY)
    if isinstance(raw_adapters, Mapping):
        adapters = dict(raw_adapters)
    else:
        adapters = {}

    raw_entry = adapters.get(adapter_key)
    if isinstance(raw_entry, Mapping):
        entry = dict(raw_entry)
    else:
        entry = {}

    entry["label"] = adapter_label

    raw_providers = entry.get("providers")
    if isinstance(raw_providers, Mapping):
        providers = dict(raw_providers)
    else:
        providers = {}

    raw_stats = providers.get(provider_key)
    if isinstance(raw_stats, Mapping):
        stats = dict(raw_stats)
    else:
        stats = {}

    stats["label"] = _PROVIDER_LABELS.get(provider_key, provider_label)
    stats["count"] = int(stats.get("count", 0) or 0) + 1

    status_counts = stats.get("status_counts")
    if isinstance(status_counts, Mapping):
        status_counts = dict(status_counts)
    else:
        status_counts = {}
    status_counts[status_key] = int(status_counts.get(status_key, 0) or 0) + 1
    stats["status_counts"] = status_counts

    if fallback:
        stats["fallback_count"] = int(stats.get("fallback_count", 0) or 0) + 1

    stats["last_status"] = status_key
    stats["fallback_last"] = bool(fallback)
    stats["ts"] = time.time()

    providers[provider_key] = stats
    entry["providers"] = providers
    adapters[adapter_key] = entry
    store[_ADAPTER_FALLBACK_KEY] = adapters

def record_quote_load(
    elapsed_ms: Optional[float], *, source: str, count: Optional[int] = None
) -> None:
    """Persist response time and source for the latest quote load."""

    store = _store()
    source_text = str(source or "unknown").strip() or "unknown"
    numeric_latency = _as_optional_float(elapsed_ms)
    numeric_count = _as_optional_int(count)
    now = time.time()

    summary: Dict[str, Any] = {
        "elapsed_ms": float(numeric_latency) if numeric_latency is not None else None,
        "source": source_text,
        "count": int(numeric_count) if numeric_count is not None else None,
        "ts": now,
    }

    total_attempts = _as_optional_int(store.get("quotes_total"))
    if total_attempts is not None:
        summary["total"] = total_attempts

    ok_attempts = _as_optional_int(store.get("quotes_ok"))
    if ok_attempts is not None:
        summary["ok"] = ok_attempts
        if total_attempts:
            summary["ok_ratio"] = ok_attempts / total_attempts

    http_counters_raw = store.get("quote_http_counters")
    if isinstance(http_counters_raw, Mapping):
        counters: Dict[str, int] = {}
        for key, value in http_counters_raw.items():
            numeric = _as_optional_int(value)
            if numeric is None or numeric <= 0:
                continue
            counters[str(key)] = numeric
        if counters:
            summary["http_counters"] = counters

    by_provider_raw = store.get("quotes_by_provider")
    if isinstance(by_provider_raw, Mapping):
        provider_summary: Dict[str, Any] = {}
        for key, value in by_provider_raw.items():
            if not isinstance(value, Mapping):
                continue
            provider_summary[str(key)] = dict(value)
        if provider_summary:
            summary["by_provider"] = provider_summary

    stats_raw = store.get("quotes_stats")
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
            stats.get("latency_history"), limit=_QUOTE_HISTORY_LIMIT
        )
        latency_history.append(value)
        stats["latency_history"] = latency_history
        stats["last_elapsed_ms"] = value
    else:
        stats["missing_latency"] = int(stats.get("missing_latency", 0) or 0) + 1

    if numeric_count is not None:
        count_value = float(numeric_count)
        stats["batch_count"] = int(stats.get("batch_count", 0) or 0) + 1
        stats["batch_sum"] = float(stats.get("batch_sum", 0.0) or 0.0) + count_value
        stats["batch_sum_sq"] = (
            float(stats.get("batch_sum_sq", 0.0) or 0.0) + count_value * count_value
        )
        current_min = _as_optional_float(stats.get("batch_min"))
        stats["batch_min"] = (
            count_value if current_min is None else min(current_min, count_value)
        )
        current_max = _as_optional_float(stats.get("batch_max"))
        stats["batch_max"] = (
            count_value if current_max is None else max(current_max, count_value)
        )
        batch_history = _ensure_latency_history(
            stats.get("batch_history"), limit=_QUOTE_HISTORY_LIMIT
        )
        batch_history.append(count_value)
        stats["batch_history"] = batch_history
        stats["last_count"] = int(numeric_count)
    else:
        stats["missing_batch"] = int(stats.get("missing_batch", 0) or 0) + 1

    stats["last_source"] = source_text
    stats["last_ts"] = now

    latest_event = dict(summary)
    event_history = _ensure_event_history(
        stats.get("event_history"), limit=_QUOTE_HISTORY_LIMIT
    )
    event_history.append(latest_event)
    stats["event_history"] = event_history

    store["quotes_stats"] = stats

    metrics_summary = _summarize_quote_stats(stats)
    if metrics_summary:
        summary["stats"] = metrics_summary

    store["quotes"] = summary

    _log_analysis_event("quotes.load", latest_event, metrics_summary)


def record_quote_provider_usage(
    provider: str,
    *,
    elapsed_ms: Optional[float],
    stale: bool,
    source: Optional[str] = None,
    http_status: Optional[str] = None,
    ok: Optional[bool] = None,
) -> None:
    """Track per-provider latency and usage statistics for quotes."""

    provider_label = str(provider or "unknown").strip() or "unknown"
    provider_key = provider_label.casefold()
    store = _store()
    raw_providers = store.get("quote_providers")
    if isinstance(raw_providers, Mapping):
        providers = dict(raw_providers)
    else:
        providers = {}

    raw_entry = providers.get(provider_key)
    if isinstance(raw_entry, Mapping):
        entry = dict(raw_entry)
    else:
        entry = {}

    entry["provider"] = provider_key
    entry["label"] = _PROVIDER_LABELS.get(provider_key, provider_label)
    total_prev = _as_optional_int(store.get("quotes_total")) or 0
    store["quotes_total"] = total_prev + 1

    entry["count"] = int(entry.get("count", 0) or 0) + 1
    if stale:
        entry["stale_count"] = int(entry.get("stale_count", 0) or 0) + 1
    entry["stale_last"] = bool(stale)
    detail_source = _clean_detail(source)
    if detail_source is not None:
        entry["last_source"] = detail_source
    elif "last_source" in entry and source is None:
        # preserve existing label when no source provided
        pass

    now = time.time()
    entry["ts"] = now

    ok_flag = bool(ok) if ok is not None else (not stale and elapsed_ms is not None)
    if ok_flag:
        entry["ok_count"] = int(entry.get("ok_count", 0) or 0) + 1
    entry["ok_last"] = ok_flag

    ok_prev = _as_optional_int(store.get("quotes_ok")) or 0
    if ok_flag:
        store["quotes_ok"] = ok_prev + 1
    else:
        store.setdefault("quotes_ok", ok_prev)

    if elapsed_ms is not None:
        elapsed_value = float(elapsed_ms)
        history_raw = entry.get("elapsed_history")
        history = _ensure_latency_history(history_raw, limit=_QUOTE_PROVIDER_HISTORY_LIMIT)
        history.append(elapsed_value)
        samples = list(history)
        entry["elapsed_history"] = samples
        entry["elapsed_last"] = elapsed_value

        percentiles = _compute_percentiles(samples, (0.5, 0.95)) if samples else {}
        if percentiles:
            p50 = percentiles.get("p50")
            p95 = percentiles.get("p95")
            if p50 is not None:
                entry["p50_ms"] = float(p50)
            if p95 is not None:
                entry["p95_ms"] = float(p95)
        else:
            entry.pop("p50_ms", None)
            entry.pop("p95_ms", None)
    else:
        entry.pop("elapsed_last", None)

    if http_status:
        status_key = str(http_status).strip()
        if status_key:
            http_raw = store.get("quote_http_counters")
            if isinstance(http_raw, Mapping):
                counters = dict(http_raw)
            else:
                counters = {}
            current = _as_optional_int(counters.get(status_key)) or 0
            counters[status_key] = current + 1
            store["quote_http_counters"] = counters
    else:
        store.setdefault("quote_http_counters", store.get("quote_http_counters", {}))

    providers[provider_key] = entry
    store["quote_providers"] = providers

    by_provider_raw = store.get("quotes_by_provider")
    if isinstance(by_provider_raw, Mapping):
        by_provider = dict(by_provider_raw)
    else:
        by_provider = {}

    provider_summary: Dict[str, Any] = {
        "provider": provider_key,
        "label": entry["label"],
        "total": int(entry.get("count", 0) or 0),
        "ok": int(entry.get("ok_count", 0) or 0),
        "stale": int(entry.get("stale_count", 0) or 0),
    }
    if "p50_ms" in entry:
        provider_summary["p50_ms"] = entry["p50_ms"]
    if "p95_ms" in entry:
        provider_summary["p95_ms"] = entry["p95_ms"]
    by_provider[provider_key] = provider_summary
    store["quotes_by_provider"] = by_provider


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


def _normalize_provider_event(entry: Any) -> Optional[Dict[str, Any]]:
    """Normalize provider history entries into a serializable mapping."""

    if not isinstance(entry, Mapping):
        return None

    provider_raw = entry.get("provider") or entry.get("source") or "unknown"
    provider_text = str(provider_raw or "unknown").strip() or "unknown"

    result_raw = (
        entry.get("result")
        or entry.get("status")
        or entry.get("latest_result")
        or entry.get("mode")
    )
    result_text = str(result_raw).strip() if result_raw is not None else None

    fallback_raw = entry.get("fallback")
    fallback_flag = bool(fallback_raw) if fallback_raw is not None else False

    ts_value = _as_optional_float(entry.get("ts"))
    detail_text = _clean_detail(entry.get("detail"))

    normalized: Dict[str, Any] = {
        "provider": provider_text,
        "fallback": fallback_flag,
    }
    if result_text:
        normalized["result"] = result_text
    if ts_value is not None:
        normalized["ts"] = ts_value
    if detail_text:
        normalized["detail"] = detail_text

    return normalized


def _compute_freshness(
    ts: Optional[float],
    *,
    now: Optional[float] = None,
    ttl: float = _SESSION_MONITORING_TTL_SECONDS,
) -> Dict[str, Any]:
    if now is None:
        now = time.time()

    if ts is None:
        return {"is_fresh": False, "age_seconds": None, "ttl_seconds": ttl}

    age = max(now - ts, 0.0)
    return {"is_fresh": age <= ttl, "age_seconds": age, "ttl_seconds": ttl, "ts": ts}


def _summarize_diagnostics(raw_entry: Any, *, now: Optional[float] = None) -> Dict[str, Any]:
    if not isinstance(raw_entry, Mapping):
        return {}

    summary: Dict[str, Any] = {}
    ts = _as_optional_float(raw_entry.get("ts"))
    snapshot_raw = raw_entry.get("snapshot")
    snapshot: Dict[str, Any] = {}
    if isinstance(snapshot_raw, Mapping):
        snapshot = {
            str(key): value
            for key, value in snapshot_raw.items()
            if str(key or "").strip()
        }

    latest: Dict[str, Any] = {}
    if snapshot:
        latest["snapshot"] = snapshot
    if ts is not None:
        latest["ts"] = ts

    source = raw_entry.get("source")
    if isinstance(source, str):
        source_text = source.strip()
        if source_text:
            latest["source"] = source_text

    if latest:
        summary["latest"] = latest

    summary["field_count"] = len(snapshot)
    summary["freshness"] = _compute_freshness(ts, now=now)
    return summary


def _summarize_dependencies(raw_entry: Any) -> Dict[str, Any]:
    if not isinstance(raw_entry, Mapping):
        return {}

    items: Dict[str, Any] = {}
    status_priority = {"critical": 3, "error": 3, "warning": 2, "degraded": 2, "ok": 1, "success": 1}
    overall_status = "unknown"
    overall_score = -1

    for name, value in raw_entry.items():
        if not isinstance(value, Mapping):
            continue
        entry: Dict[str, Any] = {}

        label_value = value.get("label") or name
        if label_value is not None:
            label_text = str(label_value).strip()
            if label_text:
                entry["label"] = label_text

        status_value = value.get("status")
        if status_value is not None:
            status_text = str(status_value).strip()
            if status_text:
                entry["status"] = status_text
                score = status_priority.get(status_text.lower(), 0)
                if score > overall_score:
                    overall_status = status_text
                    overall_score = score

        ts_value = _as_optional_float(value.get("ts"))
        if ts_value is not None:
            entry["ts"] = ts_value

        detail_text = _clean_detail(value.get("detail"))
        if detail_text:
            entry["detail"] = detail_text

        source_value = value.get("source")
        if isinstance(source_value, str) and source_value.strip():
            entry["source"] = source_value.strip()

        if entry:
            items[str(name)] = entry

    if not items:
        return {}

    summary: Dict[str, Any] = {"items": items}
    if overall_status != "unknown":
        summary["status"] = overall_status

    return summary


def _summarize_session_monitoring(
    raw_monitoring: Any, *, now: Optional[float] = None
) -> Dict[str, Any]:
    if not isinstance(raw_monitoring, Mapping):
        return {}

    if now is None:
        now = time.time()
    summary: Dict[str, Any] = {}

    active_raw = raw_monitoring.get(_ACTIVE_SESSIONS_KEY)
    sessions: list[Dict[str, Any]] = []
    latest_session_ts: Optional[float] = None
    if isinstance(active_raw, Mapping):
        for key, value in active_raw.items():
            if not isinstance(value, Mapping):
                continue
            entry: Dict[str, Any] = {
                "session_id": str(value.get("session_id") or key)
            }
            session_ts = _as_optional_float(value.get("ts"))
            if session_ts is not None:
                entry["ts"] = session_ts
                latest_session_ts = (
                    session_ts
                    if latest_session_ts is None
                    else max(latest_session_ts, session_ts)
                )
            metadata_raw = value.get("metadata")
            if isinstance(metadata_raw, Mapping):
                metadata = _normalize_metadata(metadata_raw)
                if metadata:
                    entry["metadata"] = metadata
            sessions.append(entry)

    active_summary: Dict[str, Any] = {"count": len(sessions)}
    if sessions:
        active_summary["sessions"] = sessions

    total_session_starts = _as_optional_int(raw_monitoring.get("total_session_starts"))
    if total_session_starts is not None and total_session_starts >= 0:
        active_summary["total_session_starts"] = total_session_starts

    active_ts = _as_optional_float(raw_monitoring.get("active_sessions_ts"))
    if active_ts is None:
        active_ts = latest_session_ts
    active_summary["freshness"] = _compute_freshness(active_ts, now=now)
    summary["active_sessions"] = active_summary

    login_raw = raw_monitoring.get(_LOGIN_TO_RENDER_STATS_KEY)
    if isinstance(login_raw, Mapping):
        count = _as_optional_int(login_raw.get("count")) or 0
        total = _as_optional_float(login_raw.get("sum")) or 0.0
        sum_sq = _as_optional_float(login_raw.get("sum_sq")) or 0.0
        login_summary: Dict[str, Any] = {"count": count, "total": total}
        avg = total / count if count else None
        if avg is not None:
            login_summary["avg"] = avg
            if count > 1:
                variance = max(sum_sq / count - avg * avg, 0.0)
                login_summary["stdev"] = math.sqrt(variance)
        last_value = _as_optional_float(login_raw.get("last_value"))
        last_ts = _as_optional_float(login_raw.get("last_ts"))
        if last_value is not None:
            last_entry: Dict[str, Any] = {"value": last_value}
            if last_ts is not None:
                last_entry["ts"] = last_ts
            session_ref = login_raw.get("last_session_id")
            if isinstance(session_ref, str):
                session_text = session_ref.strip()
                if session_text:
                    last_entry["session_id"] = session_text
            login_summary["last"] = last_entry
        login_summary["freshness"] = _compute_freshness(last_ts, now=now)
        summary["login_to_render"] = login_summary

    http_count = _as_optional_int(raw_monitoring.get("http_error_count")) or 0
    http_raw = raw_monitoring.get(_LAST_HTTP_ERROR_KEY)
    if isinstance(http_raw, Mapping) or http_count:
        error_summary: Dict[str, Any] = {"count": http_count}
        last_ts = None
        if isinstance(http_raw, Mapping):
            last_error: Dict[str, Any] = {}
            status_code = _as_optional_int(http_raw.get("status_code"))
            if status_code is not None:
                last_error["status_code"] = status_code
            method = http_raw.get("method")
            if isinstance(method, str) and method.strip():
                last_error["method"] = method.strip()
            url = http_raw.get("url")
            if isinstance(url, str) and url.strip():
                last_error["url"] = url.strip()
            detail = http_raw.get("detail")
            if isinstance(detail, str) and detail.strip():
                last_error["detail"] = detail.strip()
            last_ts = _as_optional_float(http_raw.get("ts"))
            if last_ts is not None:
                last_error["ts"] = last_ts
            if last_error:
                error_summary["last"] = last_error
        error_summary["freshness"] = _compute_freshness(last_ts, now=now)
        summary["http_errors"] = error_summary

    return summary


def _normalize_risk_entry(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, Mapping):
        return None

    category = entry.get("category") or entry.get("label") or "unknown"
    category_text = str(category).strip() or "unknown"

    severity_value = entry.get("severity") or entry.get("severity_label") or "unknown"
    severity_text = str(severity_value).strip() or "unknown"
    severity_key = severity_text.casefold() or "unknown"

    ts_value = _as_optional_float(entry.get("ts"))
    detail_text = _clean_detail(entry.get("detail"))

    fallback_raw = entry.get("fallback")
    fallback_flag = None if fallback_raw is None else bool(fallback_raw)

    source_text = _clean_detail(entry.get("source"))

    tags_list: Optional[list[str]] = None
    raw_tags = entry.get("tags")
    if isinstance(raw_tags, Iterable) and not isinstance(
        raw_tags, (str, bytes, bytearray)
    ):
        collected: list[str] = []
        for tag in raw_tags:
            text = str(tag).strip()
            if text:
                collected.append(text)
        if collected:
            tags_list = collected

    metadata_payload: Optional[Dict[str, Any]] = None
    raw_metadata = entry.get("metadata")
    if isinstance(raw_metadata, Mapping):
        metadata_payload = {}
        for key, value in raw_metadata.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            metadata_payload[key_text] = value
        if not metadata_payload:
            metadata_payload = None

    normalized: Dict[str, Any] = {
        "category": category_text,
        "severity": severity_key,
        "severity_label": severity_text,
    }
    if ts_value is not None:
        normalized["ts"] = ts_value
    if detail_text:
        normalized["detail"] = detail_text
    if fallback_flag is not None:
        normalized["fallback"] = fallback_flag
    if source_text:
        normalized["source"] = source_text
    if tags_list:
        normalized["tags"] = tags_list
    if metadata_payload is not None:
        normalized["metadata"] = metadata_payload

    return normalized


def _serialize_provider_history(raw_history: Any) -> list[Dict[str, Any]]:
    if not raw_history:
        return []

    if isinstance(raw_history, deque):
        iterable = list(raw_history)
    elif isinstance(raw_history, Iterable) and not isinstance(
        raw_history, (str, bytes, bytearray)
    ):
        iterable = list(raw_history)
    else:
        return []

    serialized: list[Dict[str, Any]] = []
    for entry in iterable:
        normalized = _normalize_provider_event(entry)
        if normalized is not None:
            serialized.append(normalized)
    return serialized[-_PROVIDER_HISTORY_LIMIT:]


def _serialize_provider_metrics(raw_metrics: Any) -> Any:
    if raw_metrics is None:
        return None
    if not isinstance(raw_metrics, Mapping):
        return raw_metrics

    data = dict(raw_metrics)

    source_raw = raw_metrics.get("source")
    if isinstance(source_raw, str):
        data["source"] = source_raw
    elif source_raw is not None:
        data["source"] = str(source_raw)

    detail_text = _clean_detail(raw_metrics.get("detail"))
    if detail_text is not None:
        data["detail"] = detail_text
    elif "detail" in data:
        del data["detail"]

    fallback_value = raw_metrics.get("fallback")
    if fallback_value is not None:
        data["fallback"] = bool(fallback_value)

    latest_provider = raw_metrics.get("latest_provider") or raw_metrics.get("source")
    if isinstance(latest_provider, str):
        data["latest_provider"] = latest_provider
    elif "latest_provider" in data:
        del data["latest_provider"]

    latest_result = (
        raw_metrics.get("latest_result")
        or raw_metrics.get("result")
        or raw_metrics.get("status")
    )
    if isinstance(latest_result, str):
        data["latest_result"] = latest_result
    elif "latest_result" in data:
        del data["latest_result"]

    data["history"] = _serialize_provider_history(raw_metrics.get("history"))

    return data


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


def _classify_latency_bucket(value: Optional[float]) -> str:
    if value is None:
        return "missing"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "missing"
    if not math.isfinite(numeric):
        return "missing"
    if numeric <= _LATENCY_FAST_THRESHOLD_MS:
        return "fast"
    if numeric <= _LATENCY_MEDIUM_THRESHOLD_MS:
        return "medium"
    return "slow"


def _increment_latency_bucket(stats: Dict[str, Any], prefix: str, value: Optional[float]) -> None:
    raw_buckets = stats.get(f"{prefix}_buckets")
    if isinstance(raw_buckets, Mapping):
        buckets = dict(raw_buckets)
    else:
        buckets = {}
    counts_raw = buckets.get("counts")
    if isinstance(counts_raw, Mapping):
        counts = dict(counts_raw)
    else:
        counts = {}
    bucket = _classify_latency_bucket(value)
    counts[bucket] = int(counts.get(bucket, 0) or 0) + 1
    buckets["counts"] = counts
    stats[f"{prefix}_buckets"] = buckets


def _compute_ratio_map(counts: Mapping[str, Any], total: int) -> Dict[str, float]:
    if not isinstance(counts, Mapping) or not total:
        return {}
    ratios: Dict[str, float] = {}
    for raw_key, raw_value in counts.items():
        key = str(raw_key).strip()
        if not key:
            continue
        numeric = _as_optional_float(raw_value)
        if numeric is None:
            continue
        ratios[key] = float(numeric) / total
    return ratios


def _compute_percentiles(
    samples: Sequence[float], points: Sequence[float]
) -> Dict[str, float]:
    if not samples:
        return {}

    ordered = sorted(float(value) for value in samples)
    if not ordered:
        return {}

    size = len(ordered)
    results: Dict[str, float] = {}
    for point in points:
        if point <= 0:
            results[f"p{int(point * 100):02d}"] = ordered[0]
            continue
        if point >= 1:
            results[f"p{int(point * 100):02d}"] = ordered[-1]
            continue
        position = (size - 1) * point
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            value = ordered[lower]
        else:
            weight = position - lower
            value = ordered[lower] * (1 - weight) + ordered[upper] * weight
        results[f"p{int(point * 100):02d}"] = value
    return results


def _aggregate_provider_overall(providers: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(providers, Mapping):
        return {}

    total = 0
    status_counts: Dict[str, int] = {}
    latency_counts: Dict[str, int] = {}
    error_total = 0
    fallback_total = 0

    for stats in providers.values():
        if not isinstance(stats, Mapping):
            continue
        count_value = stats.get("count") or stats.get("total")
        count = _as_optional_int(count_value)
        if count is None or count <= 0:
            continue
        total += count

        raw_status = stats.get("status_counts")
        if isinstance(raw_status, Mapping):
            for key, value in raw_status.items():
                try:
                    status_counts[str(key)] = status_counts.get(str(key), 0) + int(value)
                except (TypeError, ValueError):
                    continue

        try:
            error_total += int(stats.get("error_count", 0) or 0)
        except (TypeError, ValueError):
            pass
        try:
            fallback_total += int(stats.get("fallback_count", 0) or 0)
        except (TypeError, ValueError):
            pass

        latency = stats.get("latency_buckets")
        if isinstance(latency, Mapping):
            counts_map = latency.get("counts")
            if isinstance(counts_map, Mapping):
                for key, value in counts_map.items():
                    try:
                        latency_counts[str(key)] = latency_counts.get(str(key), 0) + int(value)
                    except (TypeError, ValueError):
                        continue

    if total <= 0:
        return {}

    return {
        "count": total,
        "status_counts": status_counts,
        "status_ratios": _compute_ratio_map(status_counts, total),
        "error_count": error_total,
        "error_ratio": error_total / total if total else 0.0,
        "fallback_count": fallback_total,
        "fallback_ratio": fallback_total / total if total else 0.0,
        "latency_buckets": {
            "counts": latency_counts,
            "ratios": _compute_ratio_map(latency_counts, total),
            "total": total,
        },
    }


def get_health_metrics() -> Dict[str, Any]:
    """Return a shallow copy of the tracked metrics for UI consumption."""
    store = _store()

    def _normalize_snapshot_event_entry(raw_event: Any) -> Dict[str, Any]:
        if not isinstance(raw_event, Mapping):
            return {}

        event: Dict[str, Any] = {}

        ts = _as_optional_float(raw_event.get("ts"))
        if ts is not None:
            event["ts"] = ts

        for key in ("kind", "status", "action"):
            value = raw_event.get(key)
            if isinstance(value, str):
                text = value.strip()
                if text:
                    event[key] = text

        detail_text = _clean_detail(raw_event.get("detail"))
        if detail_text:
            event["detail"] = detail_text

        storage_id = raw_event.get("storage_id")
        if storage_id is not None:
            text = str(storage_id).strip()
            if text:
                event["storage_id"] = text

        backend = _normalize_backend_details(raw_event.get("backend"))
        if backend:
            event["backend"] = backend

        return event

    def _normalize_diagnostics_entry(raw_entry: Any) -> Dict[str, Any]:
        if not isinstance(raw_entry, Mapping):
            return {}

        entry: Dict[str, Any] = {}

        status_value = raw_entry.get("status")
        if status_value is not None:
            status_text = str(status_value).strip()
            if status_text:
                entry["status"] = status_text

        latency_value = _as_optional_float(raw_entry.get("latency"))
        if latency_value is not None:
            entry["latency"] = latency_value

        ts_value = _as_optional_float(raw_entry.get("ts"))
        if ts_value is None:
            ts_value = _as_optional_float(raw_entry.get("timestamp"))
        if ts_value is not None:
            entry["ts"] = ts_value

        component_value = raw_entry.get("component")
        if component_value is not None:
            component_text = str(component_value).strip()
            if component_text:
                entry["component"] = component_text

        message_value = raw_entry.get("message")
        if message_value is not None:
            message_text = _clean_detail(message_value)
            if message_text:
                entry["message"] = message_text

        checks_raw = raw_entry.get("checks")
        if isinstance(checks_raw, Iterable) and not isinstance(
            checks_raw, (str, bytes, bytearray)
        ):
            checks: list[Dict[str, Any]] = []
            for item in checks_raw:
                if not isinstance(item, Mapping):
                    continue
                check_entry: Dict[str, Any] = {}
                component_name = item.get("component") or item.get("name")
                if component_name is not None:
                    component_text = str(component_name).strip()
                    if component_text:
                        check_entry["component"] = component_text
                status_entry = item.get("status")
                if status_entry is not None:
                    status_text = str(status_entry).strip()
                    if status_text:
                        check_entry["status"] = status_text
                message_entry = item.get("message")
                detail_text = _clean_detail(message_entry)
                if detail_text:
                    check_entry["message"] = detail_text
                if check_entry:
                    checks.append(check_entry)
            if checks:
                entry["checks"] = checks

        return entry
    def _merge_entry(entry: Any, stats_summary: Dict[str, Any]) -> Any:
        if not stats_summary:
            if isinstance(entry, Mapping):
                return dict(entry)
            return entry
        if isinstance(entry, Mapping):
            merged = dict(entry)
        elif entry is None:
            merged = {}
        else:
            merged = {"value": entry}
        merged["stats"] = stats_summary
        return merged

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

        def _summarize_buckets(prefix: str) -> Optional[Dict[str, Any]]:
            raw_buckets = raw_stats.get(f"{prefix}_buckets")
            if not isinstance(raw_buckets, Mapping):
                return None

            counts: Dict[str, int] = {}
            total = 0
            for bucket in ("fast", "medium", "slow", "missing"):
                try:
                    count = int(raw_buckets.get(bucket, 0) or 0)
                except (TypeError, ValueError):
                    continue
                if count < 0:
                    continue
                counts[bucket] = count
                total += count

            if total <= 0:
                return None

            ratios = {name: count / total for name, count in counts.items()}
            return {"counts": counts, "total": total, "ratios": ratios}

        elapsed_buckets = _summarize_buckets("elapsed")
        if elapsed_buckets:
            summary["elapsed_buckets"] = elapsed_buckets

        cached_buckets = _summarize_buckets("cached")
        if cached_buckets:
            summary["cached_buckets"] = cached_buckets

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

        error_total = int(raw_stats.get("error_count", 0) or 0)
        if error_total > 0:
            base_total = int(raw_stats.get("invocation_count", 0) or 0)
            ratio = (error_total / base_total) if base_total else None
            error_summary: Dict[str, Any] = {"total": error_total}
            if ratio is not None:
                error_summary["ratio"] = ratio
            error_by_mode_raw = raw_stats.get("error_by_mode")
            if isinstance(error_by_mode_raw, Mapping):
                error_summary["by_mode"] = {
                    str(name): int(value)
                    for name, value in error_by_mode_raw.items()
                    if _as_optional_int(value) is not None
                }
            summary["errors"] = error_summary

        return summary

    def _summarize_tab_latencies(raw_tabs: Any) -> Dict[str, Any]:
        if not isinstance(raw_tabs, Mapping):
            return {}

        summary: Dict[str, Any] = {}
        for key, raw_stats in raw_tabs.items():
            if not isinstance(raw_stats, Mapping):
                continue

            label = str(raw_stats.get("label") or key).strip() or str(key)
            total_attempts = _as_optional_int(raw_stats.get("total"))
            if total_attempts is None:
                total_attempts = 0

            latency_count = _as_optional_int(raw_stats.get("count")) or 0
            sum_value = _as_optional_float(raw_stats.get("sum")) or 0.0
            sum_sq_value = _as_optional_float(raw_stats.get("sum_sq")) or 0.0
            avg = sum_value / latency_count if latency_count else None
            stdev = None
            if latency_count and latency_count > 1:
                variance = max(sum_sq_value / latency_count - (avg or 0.0) ** 2, 0.0)
                stdev = math.sqrt(variance)

            history_raw = raw_stats.get("history")
            samples: list[float] = []
            if isinstance(history_raw, deque):
                samples = [float(item) for item in history_raw]
            elif isinstance(history_raw, Iterable) and not isinstance(
                history_raw, (str, bytes, bytearray)
            ):
                for value in history_raw:
                    numeric = _as_optional_float(value)
                    if numeric is None:
                        continue
                    samples.append(float(numeric))

            percentiles = (
                _compute_percentiles(samples, (0.5, 0.9, 0.95, 0.99)) if samples else {}
            )

            status_counts_raw = raw_stats.get("status_counts")
            status_counts: Dict[str, int] = {}
            if isinstance(status_counts_raw, Mapping):
                for status_key, value in status_counts_raw.items():
                    numeric = _as_optional_float(value)
                    if numeric is None:
                        continue
                    status_counts[str(status_key)] = int(numeric)

            error_count = _as_optional_int(raw_stats.get("error_count")) or 0
            error_ratio = error_count / total_attempts if total_attempts else 0.0

            summary[key] = {
                "label": label,
                "count": latency_count,
                "total": total_attempts,
                "avg": avg,
                "stdev": stdev,
                "percentiles": percentiles,
                "status_counts": status_counts,
                "status_ratios": _compute_ratio_map(status_counts, total_attempts),
                "error_count": error_count,
                "error_ratio": error_ratio,
                "error_budget": error_ratio,
                "missing_count": _as_optional_int(raw_stats.get("missing_count")) or 0,
                "latest": {
                    "elapsed_ms": _as_optional_float(raw_stats.get("last_elapsed_ms")),
                    "status": raw_stats.get("last_status"),
                    "ts": _as_optional_float(raw_stats.get("ts")),
                },
            }

        return summary

    def _summarize_quote_providers(
        raw_providers: Any, raw_rate_limits: Any
    ) -> Dict[str, Any]:
        if not isinstance(raw_providers, Mapping):
            return {}

        store = _store()
        providers: list[Dict[str, Any]] = []
        total_count = 0
        stale_total = 0
        ok_total = 0
        rate_total = 0
        rate_wait_total = 0.0

        rate_limit_map: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_rate_limits, Mapping):
            for rate_key, rate_entry in raw_rate_limits.items():
                if isinstance(rate_entry, Mapping):
                    rate_limit_map[str(rate_key).casefold()] = dict(rate_entry)

        for key, entry in raw_providers.items():
            if not isinstance(entry, Mapping):
                continue

            count = _as_optional_int(entry.get("count"))
            if count is None or count <= 0:
                continue

            stale_count = _as_optional_int(entry.get("stale_count")) or 0
            total_count += count
            stale_total += max(stale_count, 0)
            ok_count = _as_optional_int(entry.get("ok_count")) or 0
            ok_total += max(ok_count, 0)

            label = str(entry.get("label") or entry.get("provider") or key)

            history_raw = entry.get("elapsed_history")
            latencies: list[float] = []
            if isinstance(history_raw, Iterable) and not isinstance(
                history_raw, (str, bytes, bytearray)
            ):
                for value in history_raw:
                    numeric = _as_optional_float(value)
                    if numeric is not None:
                        latencies.append(numeric)

            avg_ms: Optional[float] = None
            if latencies:
                avg_ms = sum(latencies) / len(latencies)
            percentiles = _compute_percentiles(latencies, (0.5, 0.95)) if latencies else {}

            provider_summary: Dict[str, Any] = {
                "provider": str(key),
                "label": label,
                "count": count,
            }
            if stale_count:
                provider_summary["stale_count"] = stale_count
            if ok_count:
                provider_summary["ok_count"] = ok_count
                provider_summary["ok_ratio"] = ok_count / count if count else 0.0

            last_ms = _as_optional_float(entry.get("elapsed_last"))
            if avg_ms is not None:
                provider_summary["avg_ms"] = avg_ms
            if last_ms is not None:
                provider_summary["last_ms"] = last_ms
            if percentiles:
                p50 = percentiles.get("p50")
                p95 = percentiles.get("p95")
                if p50 is not None:
                    provider_summary["p50_ms"] = float(p50)
                if p95 is not None:
                    provider_summary["p95_ms"] = float(p95)

            ts_value = _as_optional_float(entry.get("ts"))
            if ts_value is not None:
                provider_summary["ts"] = ts_value

            source_detail = _clean_detail(entry.get("last_source"))
            if source_detail:
                provider_summary["source"] = source_detail

            if entry.get("stale_last"):
                provider_summary["stale_last"] = True

            rate_entry = rate_limit_map.get(str(key).casefold())
            if isinstance(rate_entry, Mapping):
                rate_count = _as_optional_int(rate_entry.get("count"))
                if rate_count:
                    provider_summary["rate_limit_count"] = rate_count
                    rate_total += rate_count
                    wait_total = _as_optional_float(rate_entry.get("wait_total")) or 0.0
                    rate_wait_total += wait_total
                    last_wait = _as_optional_float(rate_entry.get("wait_last"))
                    if last_wait is not None:
                        provider_summary["rate_limit_last_ms"] = last_wait * 1000.0
                    if wait_total and rate_count:
                        provider_summary["rate_limit_avg_ms"] = (
                            wait_total / rate_count
                        ) * 1000.0
                    last_reason = rate_entry.get("last_reason")
                    if last_reason:
                        provider_summary["rate_limit_last_reason"] = str(last_reason)
                    limit_ts = _as_optional_float(rate_entry.get("ts"))
                    if limit_ts is not None:
                        provider_summary["rate_limit_ts"] = limit_ts

            providers.append(provider_summary)

        if not providers:
            return {}

        providers.sort(key=lambda item: str(item.get("label", "")).casefold())
        summary_total = _as_optional_int(store.get("quotes_total")) or total_count
        summary: Dict[str, Any] = {"providers": providers, "total": summary_total}
        if stale_total:
            summary["stale_total"] = stale_total
        if ok_total:
            summary["ok_total"] = ok_total
        if rate_total:
            summary["rate_limit_total"] = rate_total
        if rate_wait_total:
            summary["rate_limit_total_wait_ms"] = rate_wait_total * 1000.0
        return summary

    def _summarize_adapter_fallbacks(raw_data: Any) -> Dict[str, Any]:
        if not isinstance(raw_data, Mapping):
            return {}

        adapters: Dict[str, Any] = {}
        provider_totals: Dict[str, Dict[str, Any]] = {}

        for adapter_key, raw_entry in raw_data.items():
            if not isinstance(raw_entry, Mapping):
                continue

            label = str(raw_entry.get("label") or adapter_key).strip() or str(adapter_key)
            providers_summary: Dict[str, Any] = {}

            raw_providers = raw_entry.get("providers")
            if not isinstance(raw_providers, Mapping):
                continue

            for provider_key, raw_stats in raw_providers.items():
                if not isinstance(raw_stats, Mapping):
                    continue

                count = _as_optional_int(raw_stats.get("count")) or 0
                fallback_count = _as_optional_int(raw_stats.get("fallback_count")) or 0
                status_counts_raw = raw_stats.get("status_counts")
                status_counts: Dict[str, int] = {}
                if isinstance(status_counts_raw, Mapping):
                    for status_name, value in status_counts_raw.items():
                        numeric = _as_optional_float(value)
                        if numeric is None:
                            continue
                        status_counts[str(status_name)] = int(numeric)

                total_attempts = count or sum(status_counts.values())
                provider_label = _PROVIDER_LABELS.get(
                    str(provider_key).casefold(),
                    str(raw_stats.get("label") or provider_key).strip() or str(provider_key),
                )

                entry = {
                    "label": provider_label,
                    "count": count,
                    "fallback_count": fallback_count,
                    "fallback_ratio": (fallback_count / total_attempts)
                    if total_attempts
                    else 0.0,
                    "status_counts": status_counts,
                    "status_ratios": _compute_ratio_map(status_counts, total_attempts),
                    "latest": {
                        "status": raw_stats.get("last_status"),
                        "fallback": bool(raw_stats.get("fallback_last")),
                        "ts": _as_optional_float(raw_stats.get("ts")),
                    },
                }

                providers_summary[str(provider_key)] = entry

                aggregate = provider_totals.setdefault(
                    str(provider_key),
                    {
                        "label": provider_label,
                        "count": 0,
                        "fallback_count": 0,
                        "status_counts": {},
                        "latest": None,
                    },
                )
                aggregate["count"] = int(aggregate.get("count", 0)) + count
                aggregate["fallback_count"] = int(
                    aggregate.get("fallback_count", 0)
                ) + fallback_count

                agg_status = aggregate.get("status_counts")
                if not isinstance(agg_status, dict):
                    agg_status = {}
                for status_name, value in status_counts.items():
                    agg_status[status_name] = int(agg_status.get(status_name, 0)) + int(value)
                aggregate["status_counts"] = agg_status

                latest_payload = aggregate.get("latest")
                current_ts = _as_optional_float(raw_stats.get("ts"))
                if current_ts is not None:
                    if not isinstance(latest_payload, Mapping):
                        aggregate["latest"] = {
                            "status": raw_stats.get("last_status"),
                            "fallback": bool(raw_stats.get("fallback_last")),
                            "ts": current_ts,
                        }
                    else:
                        existing_ts = _as_optional_float(latest_payload.get("ts"))
                        if existing_ts is None or current_ts >= existing_ts:
                            aggregate["latest"] = {
                                "status": raw_stats.get("last_status"),
                                "fallback": bool(raw_stats.get("fallback_last")),
                                "ts": current_ts,
                            }

            if providers_summary:
                adapters[str(adapter_key)] = {"label": label, "providers": providers_summary}

        for provider_key, aggregate in provider_totals.items():
            total_attempts = int(aggregate.get("count", 0))
            status_counts = aggregate.get("status_counts")
            if not isinstance(status_counts, dict):
                status_counts = {}
            aggregate["status_ratios"] = _compute_ratio_map(status_counts, total_attempts)
            fallback_count = int(aggregate.get("fallback_count", 0))
            aggregate["fallback_ratio"] = (
                fallback_count / total_attempts if total_attempts else 0.0
            )

        return {"adapters": adapters, "providers": provider_totals}

    def _summarize_risk(raw_risk: Any) -> Dict[str, Any]:
        if not isinstance(raw_risk, Mapping):
            return {}

        summary: Dict[str, Any] = {}

        total = _as_optional_int(raw_risk.get("total"))
        if total is not None and total >= 0:
            summary["total"] = total

        fallback_total = _as_optional_int(raw_risk.get("fallback_count"))
        if fallback_total is not None and fallback_total >= 0:
            summary["fallback_count"] = fallback_total
            if total:
                summary["fallback_ratio"] = fallback_total / total

        raw_severities = raw_risk.get("by_severity")
        severities_summary: Dict[str, Any] = {}
        if isinstance(raw_severities, Mapping):
            for key, raw_stats in raw_severities.items():
                if not isinstance(raw_stats, Mapping):
                    continue
                count = _as_optional_int(raw_stats.get("count")) or 0
                entry: Dict[str, Any] = {"count": count}
                label = raw_stats.get("label")
                if isinstance(label, str) and label.strip():
                    entry["label"] = label.strip()
                fallback_count = _as_optional_int(raw_stats.get("fallback_count")) or 0
                if fallback_count:
                    entry["fallback_count"] = fallback_count
                if count:
                    entry["fallback_ratio"] = fallback_count / count
                categories_raw = raw_stats.get("categories")
                if isinstance(categories_raw, Mapping):
                    categories_counts: Dict[str, int] = {}
                    total_categories = 0
                    for cat_key, value in categories_raw.items():
                        numeric = _as_optional_int(value)
                        if numeric is None or numeric < 0:
                            continue
                        categories_counts[str(cat_key)] = numeric
                        total_categories += numeric
                    if categories_counts:
                        entry["categories"] = categories_counts
                        entry["category_ratios"] = _compute_ratio_map(
                            categories_counts, total_categories
                        )
                last_ts = _as_optional_float(raw_stats.get("last_ts"))
                if last_ts is not None:
                    entry["last_ts"] = last_ts
                last_category = raw_stats.get("last_category")
                if isinstance(last_category, str) and last_category.strip():
                    entry["last_category"] = last_category.strip()
                severities_summary[str(key)] = entry
        if severities_summary:
            summary["by_severity"] = severities_summary

        raw_categories = raw_risk.get("by_category")
        categories_summary: Dict[str, Any] = {}
        if isinstance(raw_categories, Mapping):
            for key, raw_stats in raw_categories.items():
                if not isinstance(raw_stats, Mapping):
                    continue
                count = _as_optional_int(raw_stats.get("count")) or 0
                entry: Dict[str, Any] = {"count": count}
                label = raw_stats.get("label")
                if isinstance(label, str) and label.strip():
                    entry["label"] = label.strip()
                fallback_count = _as_optional_int(raw_stats.get("fallback_count")) or 0
                if fallback_count:
                    entry["fallback_count"] = fallback_count
                if count:
                    entry["fallback_ratio"] = fallback_count / count
                severity_counts_raw = raw_stats.get("severity_counts")
                if isinstance(severity_counts_raw, Mapping):
                    severity_counts: Dict[str, int] = {}
                    for severity_name, value in severity_counts_raw.items():
                        numeric = _as_optional_int(value)
                        if numeric is None or numeric < 0:
                            continue
                        severity_counts[str(severity_name)] = numeric
                    if severity_counts:
                        entry["severity_counts"] = severity_counts
                        entry["severity_ratios"] = _compute_ratio_map(
                            severity_counts, count
                        )
                last_ts = _as_optional_float(raw_stats.get("last_ts"))
                if last_ts is not None:
                    entry["last_ts"] = last_ts
                last_detail = _clean_detail(raw_stats.get("last_detail"))
                if last_detail:
                    entry["last_detail"] = last_detail
                last_severity = raw_stats.get("last_severity")
                if isinstance(last_severity, str) and last_severity.strip():
                    entry["last_severity"] = last_severity.strip()
                last_fallback = raw_stats.get("last_fallback")
                if last_fallback is not None:
                    entry["last_fallback"] = bool(last_fallback)
                last_source = _clean_detail(raw_stats.get("last_source"))
                if last_source:
                    entry["last_source"] = last_source
                last_tags = raw_stats.get("last_tags")
                if isinstance(last_tags, Iterable) and not isinstance(
                    last_tags, (str, bytes, bytearray)
                ):
                    collected_tags = [
                        str(tag).strip()
                        for tag in last_tags
                        if str(tag).strip()
                    ]
                    if collected_tags:
                        entry["last_tags"] = collected_tags
                categories_summary[str(key)] = entry
        if categories_summary:
            summary["by_category"] = categories_summary

        latest = raw_risk.get("latest")
        normalized_latest = _normalize_risk_entry(latest)
        if normalized_latest:
            summary["latest"] = normalized_latest

        raw_latest_by_category = raw_risk.get("latest_by_category")
        if isinstance(raw_latest_by_category, Mapping):
            latest_by_category: Dict[str, Any] = {}
            for key, value in raw_latest_by_category.items():
                normalized_entry = _normalize_risk_entry(value)
                if normalized_entry:
                    latest_by_category[str(key)] = normalized_entry
            if latest_by_category:
                summary["latest_by_category"] = latest_by_category

        history_raw = raw_risk.get("history")
        history_entries: list[Dict[str, Any]] = []
        if isinstance(history_raw, deque):
            iterable_history = list(history_raw)
        elif isinstance(history_raw, Iterable) and not isinstance(
            history_raw, (str, bytes, bytearray)
        ):
            iterable_history = list(history_raw)
        else:
            iterable_history = []
        for entry in iterable_history:
            normalized_entry = _normalize_risk_entry(entry)
            if normalized_entry:
                history_entries.append(normalized_entry)
        if history_entries:
            summary["history"] = history_entries

        return summary

    def _summarize_macro(raw_macro: Any) -> Dict[str, Any]:
        if not isinstance(raw_macro, Mapping):
            return {}

        summary: Dict[str, Any] = {}

        latest = raw_macro.get("latest")
        if isinstance(latest, Mapping):
            summary["latest"] = dict(latest)

        raw_providers = raw_macro.get("providers")
        providers: Dict[str, Any] = {}
        overall_status_counts: Dict[str, int] = {}
        overall_buckets: Dict[str, int] = {}
        overall_total = 0
        overall_errors = 0
        overall_fallbacks = 0

        if isinstance(raw_providers, Mapping):
            for key, raw_stats in raw_providers.items():
                provider_name = str(key)
                if not isinstance(raw_stats, Mapping):
                    continue

                provider_summary: Dict[str, Any] = {}
                latest_entry = raw_stats.get("latest")
                if isinstance(latest_entry, Mapping):
                    provider_summary["latest"] = dict(latest_entry)

                total_count = _as_optional_int(raw_stats.get("total")) or 0
                provider_summary["count"] = total_count
                overall_total += total_count

                label = raw_stats.get("label")
                if isinstance(label, str) and label:
                    provider_summary["label"] = label
                else:
                    provider_summary["label"] = provider_name

                raw_status_counts = raw_stats.get("status_counts")
                status_counts: Dict[str, int] = {}
                if isinstance(raw_status_counts, Mapping):
                    for status_name, value in raw_status_counts.items():
                        count = _as_optional_int(value)
                        if count is None or count < 0:
                            continue
                        status_key = str(status_name)
                        status_counts[status_key] = count
                        overall_status_counts[status_key] = (
                            overall_status_counts.get(status_key, 0) + count
                        )
                if status_counts:
                    provider_summary["status_counts"] = status_counts
                    provider_summary["status_ratios"] = {
                        status: count / total_count
                        for status, count in status_counts.items()
                        if total_count
                    }

                fallback_count = _as_optional_int(raw_stats.get("fallback_count")) or 0
                provider_summary["fallback_count"] = fallback_count
                if total_count:
                    provider_summary["fallback_ratio"] = fallback_count / total_count
                overall_fallbacks += fallback_count

                error_count = _as_optional_int(raw_stats.get("error_count")) or 0
                provider_summary["error_count"] = error_count
                if total_count:
                    provider_summary["error_ratio"] = error_count / total_count
                overall_errors += error_count

                raw_history = raw_stats.get("history")
                if isinstance(raw_history, Iterable) and not isinstance(
                    raw_history, (str, bytes, bytearray)
                ):
                    history_entries: list[Dict[str, Any]] = []
                    for entry in raw_history:
                        if isinstance(entry, Mapping):
                            history_entries.append(dict(entry))
                    if history_entries:
                        provider_summary["history"] = history_entries

                raw_buckets = raw_stats.get("latency_buckets")
                bucket_counts: Dict[str, int] = {}
                bucket_total = 0
                if isinstance(raw_buckets, Mapping):
                    counts_section = raw_buckets.get("counts")
                    if isinstance(counts_section, Mapping):
                        source_counts: Mapping[str, Any] = counts_section
                    else:
                        source_counts = raw_buckets
                    for bucket in ("fast", "medium", "slow", "missing"):
                        count = _as_optional_int(source_counts.get(bucket))
                        if count is None or count < 0:
                            continue
                        bucket_counts[bucket] = count
                        bucket_total += count
                        overall_buckets[bucket] = overall_buckets.get(bucket, 0) + count
                if bucket_counts and bucket_total:
                    provider_summary["latency_buckets"] = {
                        "counts": bucket_counts,
                        "total": bucket_total,
                        "ratios": {
                            name: count / bucket_total
                            for name, count in bucket_counts.items()
                            if bucket_total
                        },
                    }

                providers[provider_name] = provider_summary

        if providers:
            summary["providers"] = providers

        if overall_total > 0:
            latency_total = sum(overall_buckets.values())
            latency_summary: Optional[Dict[str, Any]] = None
            if latency_total > 0:
                latency_summary = {
                    "counts": dict(overall_buckets),
                    "total": latency_total,
                    "ratios": {
                        name: count / latency_total
                        for name, count in overall_buckets.items()
                        if latency_total
                    },
                }

            summary["overall"] = {
                "count": overall_total,
                "status_counts": overall_status_counts,
                "status_ratios": {
                    name: count / overall_total
                    for name, count in overall_status_counts.items()
                    if overall_total
                },
                "fallback_count": overall_fallbacks,
                "fallback_ratio": overall_fallbacks / overall_total,
                "error_count": overall_errors,
                "error_ratio": overall_errors / overall_total,
            }
            if latency_summary:
                summary["overall"]["latency_buckets"] = latency_summary

        return summary

    now = time.time()
    diagnostics_data = _summarize_diagnostics(
        store.get(_DIAGNOSTICS_SNAPSHOT_KEY), now=now
    )
    session_monitoring_data = _summarize_session_monitoring(
        store.get(_SESSION_MONITORING_KEY), now=now
    )

    fx_metrics = fx_metrics_snapshot(store)
    portfolio_metrics = portfolio_metrics_snapshot(store)
    fx_api_data = fx_metrics.get("fx_api")
    fx_cache_data = fx_metrics.get("fx_cache")
    portfolio_data = portfolio_metrics.get("portfolio")
    quotes_data = _merge_entry(
        store.get("quotes"), _summarize_quote_stats(store.get("quotes_stats"))
    )

    return {
        "iol_refresh": store.get("iol_refresh"),
        "snapshot_event": _normalize_snapshot_event_entry(store.get(_SNAPSHOT_EVENT_KEY)),
        "diagnostics": diagnostics_data,
        "session_monitoring": session_monitoring_data,
        "startup_diagnostics": _normalize_diagnostics_entry(
            store.get(_DIAGNOSTICS_SNAPSHOT_KEY)
        ),
        "yfinance": _serialize_provider_metrics(store.get("yfinance")),
        "market_data": list(store.get(_MARKET_DATA_INCIDENTS_KEY, [])),
        "risk_incidents": _summarize_risk(store.get(_RISK_INCIDENTS_KEY)),
        "fx_api": fx_api_data,
        "fx_cache": fx_cache_data,
        "macro_api": _summarize_macro(store.get("macro_api")),
        "portfolio": portfolio_data,
        "quotes": quotes_data,
        "quote_providers": _summarize_quote_providers(
            store.get("quote_providers"), store.get(_QUOTE_RATE_LIMIT_KEY)
        ),
        "tab_latencies": _summarize_tab_latencies(store.get(_TAB_LATENCIES_KEY)),
        "adapter_fallbacks": _summarize_adapter_fallbacks(store.get(_ADAPTER_FALLBACK_KEY)),
        "dependencies": _summarize_dependencies(store.get(_DEPENDENCIES_KEY)),
    }


__all__ = [
    "get_health_metrics",
    "record_diagnostics_snapshot",
    "record_dependency_status",
    "record_fx_api_response",
    "record_fx_cache_usage",
    "record_http_error",
    "record_iol_refresh",
    "record_login_to_render",
    "record_portfolio_load",
    "record_environment_snapshot",
    "record_session_started",
    "record_tab_latency",
    "record_adapter_fallback",
    "record_quote_load",
    "record_quote_provider_usage",
    "record_quote_rate_limit_wait",
    "record_snapshot_event",
    "record_market_data_incident",
    "record_risk_incident",
    "record_yfinance_usage",
]
