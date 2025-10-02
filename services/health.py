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
_LATENCY_FAST_THRESHOLD_MS = 250.0
_LATENCY_MEDIUM_THRESHOLD_MS = 750.0


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
    provider_label = str(provider or "unknown").strip() or "unknown"
    provider_key = provider_label.casefold()
    status_value = str(status or "unknown").strip().casefold() or "unknown"
    elapsed_value = _as_optional_float(elapsed_ms)
    latest_entry = {
        "provider": provider_key,
        "provider_label": provider_label,
        "status": status_value,
        "elapsed_ms": elapsed_value,
        "detail": _clean_detail(detail),
        "fallback": bool(fallback),
        "ts": time.time(),
    }

    raw_macro = store.get("macro_api")
    macro_data: Dict[str, Any]
    if isinstance(raw_macro, Mapping):
        macro_data = dict(raw_macro)
    else:
        macro_data = {}

    macro_data["latest"] = latest_entry

    providers: Dict[str, Any]
    raw_providers = macro_data.get("providers")
    if isinstance(raw_providers, Mapping):
        providers = dict(raw_providers)
    else:
        providers = {}

    provider_stats_raw = providers.get(provider_key)
    if isinstance(provider_stats_raw, Mapping):
        provider_stats = dict(provider_stats_raw)
    else:
        provider_stats = {}

    provider_stats["latest"] = latest_entry
    provider_stats["label"] = provider_label
    provider_stats["total"] = int(provider_stats.get("total", 0) or 0) + 1

    status_counts = provider_stats.get("status_counts")
    if not isinstance(status_counts, dict):
        status_counts = {}
    status_counts[status_value] = int(status_counts.get(status_value, 0) or 0) + 1
    provider_stats["status_counts"] = status_counts

    latency_buckets = provider_stats.get("latency_buckets")
    if not isinstance(latency_buckets, dict):
        latency_buckets = {}
    bucket = _classify_latency_bucket(elapsed_value)
    latency_buckets[bucket] = int(latency_buckets.get(bucket, 0) or 0) + 1
    provider_stats["latency_buckets"] = latency_buckets

    if fallback:
        provider_stats["fallback_count"] = int(
            provider_stats.get("fallback_count", 0) or 0
        ) + 1

    if status_value == "error":
        provider_stats["error_count"] = int(
            provider_stats.get("error_count", 0) or 0
        ) + 1

    providers[provider_key] = provider_stats
    macro_data["providers"] = providers
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
    bucket = _classify_latency_bucket(value)
    buckets[bucket] = int(buckets.get(bucket, 0) or 0) + 1
    stats[f"{prefix}_buckets"] = buckets


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

                raw_buckets = raw_stats.get("latency_buckets")
                if isinstance(raw_buckets, Mapping):
                    bucket_counts: Dict[str, int] = {}
                    bucket_total = 0
                    for bucket in ("fast", "medium", "slow", "missing"):
                        count = _as_optional_int(raw_buckets.get(bucket))
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
        "yfinance": store.get("yfinance"),
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
    "record_yfinance_usage",
]
