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
_MARKET_DATA_INCIDENTS_KEY = "market_data_incidents"
_MARKET_DATA_INCIDENT_LIMIT = 20
_LATENCY_FAST_THRESHOLD_MS = 250.0
_LATENCY_MEDIUM_THRESHOLD_MS = 750.0
_PROVIDER_HISTORY_LIMIT = 8


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

    stats["invocation_count"] = int(stats.get("invocation_count", 0) or 0) + 1

    modes: Dict[str, int] = stats.setdefault("modes", {})
    modes[mode] = int(modes.get(mode, 0)) + 1

    elapsed_value = _as_optional_float(elapsed_ms)
    if elapsed_value is not None:
        stats["elapsed_count"] = int(stats.get("elapsed_count", 0)) + 1
        stats["elapsed_sum"] = float(stats.get("elapsed_sum", 0.0)) + elapsed_value
        stats["elapsed_sum_sq"] = (
            float(stats.get("elapsed_sum_sq", 0.0)) + elapsed_value * elapsed_value
        )
        _increment_latency_bucket(stats, "elapsed", elapsed_value)
    else:
        _increment_latency_bucket(stats, "elapsed", None)

    cached_value = _as_optional_float(cached_elapsed_ms)
    if cached_value is not None:
        stats["cached_count"] = int(stats.get("cached_count", 0)) + 1
        stats["cached_sum"] = float(stats.get("cached_sum", 0.0)) + cached_value
        stats["cached_sum_sq"] = (
            float(stats.get("cached_sum_sq", 0.0)) + cached_value * cached_value
        )
        _increment_latency_bucket(stats, "cached", cached_value)
    else:
        _increment_latency_bucket(stats, "cached", None)

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

    if str(mode or "").casefold() in {"error", "failure"}:
        stats["error_count"] = int(stats.get("error_count", 0) or 0) + 1
        by_mode = stats.setdefault("error_by_mode", {})
        by_mode[mode] = int(by_mode.get(mode, 0) or 0) + 1

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

    return {
        "iol_refresh": store.get("iol_refresh"),
        "yfinance": _serialize_provider_metrics(store.get("yfinance")),
        "market_data": list(store.get(_MARKET_DATA_INCIDENTS_KEY, [])),
        "fx_api": store.get("fx_api"),
        "fx_cache": store.get("fx_cache"),
        "macro_api": _summarize_macro(store.get("macro_api")),
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
    "record_market_data_incident",
    "record_yfinance_usage",
]
