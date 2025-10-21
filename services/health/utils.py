"""Utility helpers shared across health modules."""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Sequence


def _clean_detail(detail: Optional[str]) -> Optional[str]:
    if detail is None:
        return None
    text = str(detail).strip()
    return text or None


def _normalize_backend_details(raw_backend: Any) -> Dict[str, Any]:
    if not isinstance(raw_backend, Mapping):
        return {}

    details: Dict[str, Any] = {}
    for key, value in raw_backend.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        if value is None:
            continue
        if isinstance(value, (str, bytes)):
            text = str(value).strip()
            if not text:
                continue
            details[key_text] = text
        elif isinstance(value, (int, float, bool)):
            details[key_text] = value
        else:
            details[key_text] = str(value)
    return details


def _normalize_metadata(raw_metadata: Any) -> Dict[str, Any]:
    if not isinstance(raw_metadata, Mapping):
        return {}

    normalized: Dict[str, Any] = {}
    for key, value in raw_metadata.items():
        key_text = str(key or "").strip()
        if not key_text:
            continue
        if value is None:
            continue
        if isinstance(value, (bool, int, float)):
            normalized[key_text] = value
        elif isinstance(value, (str, bytes)):
            text = str(value).strip()
            if text:
                normalized[key_text] = text
        else:
            normalized[key_text] = str(value)
    return normalized


def _normalize_environment_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        normalized_map: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key or "").strip()
            if not key_text:
                continue
            normalized_item = _normalize_environment_value(item)
            if normalized_item is None:
                continue
            if isinstance(normalized_item, Mapping) and not normalized_item:
                continue
            if isinstance(normalized_item, (list, tuple)) and not normalized_item:
                continue
            normalized_map[key_text] = normalized_item
        return normalized_map
    if isinstance(value, (list, tuple, set)):
        normalized_list = [
            item
            for item in (
                _normalize_environment_value(entry)
                for entry in value
            )
            if item is not None and (not isinstance(item, (list, tuple, Mapping)) or item)
        ]
        return normalized_list
    return str(value)


def _normalize_environment_snapshot(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in snapshot.items():
        key_text = str(key or "").strip()
        if not key_text:
            continue
        normalized_value = _normalize_environment_value(value)
        if normalized_value is None:
            continue
        if isinstance(normalized_value, Mapping) and not normalized_value:
            continue
        if isinstance(normalized_value, (list, tuple)) and not normalized_value:
            continue
        normalized[key_text] = normalized_value
    return normalized


def _ensure_history_deque(
    history: Any, *, limit: int = 32
) -> Deque[Mapping[str, Any]]:
    if isinstance(history, deque):
        return deque(history, maxlen=limit)
    if isinstance(history, Iterable):
        return deque(history, maxlen=limit)
    return deque(maxlen=limit)


def _ensure_event_history(
    history: Any, *, limit: int = 32
) -> Deque[Mapping[str, Any]]:
    return _ensure_history_deque(history, limit=limit)


def _ensure_latency_history(history: Any, *, limit: int = 32) -> Deque[float]:
    if isinstance(history, deque):
        return deque(history, maxlen=limit)
    if isinstance(history, Iterable):
        return deque((float(value) for value in history), maxlen=limit)
    return deque(maxlen=limit)


def _merge_entry(
    existing: Any, summary: Mapping[str, Any] | None
) -> Dict[str, Any] | None:
    if not isinstance(existing, Mapping) and not summary:
        return None

    merged: Dict[str, Any] = {}
    if isinstance(existing, Mapping):
        merged.update(existing)
    if summary:
        merged.update(summary)
    return merged or None


def _compute_ratio_map(
    counts: Mapping[str, int], total: int
) -> Dict[str, float]:
    if total <= 0:
        return {key: 0.0 for key in counts}
    return {key: value / total for key, value in counts.items()}


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


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
        return None
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return int(value)
        return None
    try:
        numeric = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return numeric


def _ensure_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return [value] if value is not None else []


__all__ = [
    "_clean_detail",
    "_normalize_backend_details",
    "_normalize_metadata",
    "_normalize_environment_value",
    "_normalize_environment_snapshot",
    "_ensure_history_deque",
    "_ensure_event_history",
    "_ensure_latency_history",
    "_merge_entry",
    "_compute_ratio_map",
    "_normalize_numeric",
    "_serialize_event_history",
    "_summarize_metric_block",
    "_normalize_counter_map",
    "_as_optional_float",
    "_as_optional_int",
    "_ensure_sequence",
]
