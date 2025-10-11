from __future__ import annotations

from typing import Mapping

import numpy as np
import streamlit as st

from shared.settings import CACHE_HIT_THRESHOLDS

__all__ = [
    "_normalise_cache_stats",
    "_resolve_cache_status_color",
    "_render_cache_status",
]


def _normalise_cache_stats(stats: object) -> dict[str, object]:
    if isinstance(stats, Mapping):
        return dict(stats)
    if hasattr(stats, "as_dict"):
        try:
            data = dict(stats.as_dict())
        except Exception:  # pragma: no cover - defensive
            data = {}
    else:
        data = {}
    for attribute in (
        "hits",
        "misses",
        "last_updated",
        "ttl_hours",
        "remaining_ttl",
        "hit_ratio",
    ):
        if attribute in data:
            continue
        if hasattr(stats, attribute):
            try:
                data[attribute] = getattr(stats, attribute)
            except Exception:  # pragma: no cover - defensive
                continue
    return data


def _resolve_cache_status_color(ratio: float) -> str:
    try:
        green_threshold = float(CACHE_HIT_THRESHOLDS.get("green", 0.7))
    except Exception:  # pragma: no cover - defensive
        green_threshold = 0.7
    try:
        yellow_threshold = float(CACHE_HIT_THRESHOLDS.get("yellow", 0.4))
    except Exception:  # pragma: no cover - defensive
        yellow_threshold = 0.4
    if ratio >= green_threshold:
        return "green"
    if ratio >= yellow_threshold:
        return "yellow"
    return "red"


def _render_cache_status(cache_stats: Mapping[str, object]) -> str:
    raw_ratio = cache_stats.get("hit_ratio", 0.0)
    try:
        ratio = float(raw_ratio)
    except (TypeError, ValueError):
        ratio = 0.0
    if not np.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        ratio = 0.0
    color = _resolve_cache_status_color(ratio)

    ttl_value = cache_stats.get("remaining_ttl")
    ttl_seconds: float | None
    if isinstance(ttl_value, (int, float)) and np.isfinite(float(ttl_value)):
        ttl_seconds = float(ttl_value)
    else:
        ttl_hours = cache_stats.get("ttl_hours")
        ttl_seconds = None
        if isinstance(ttl_hours, (int, float)):
            try:
                ttl_seconds = max(float(ttl_hours) * 3600.0, 0.0)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                ttl_seconds = None
    ttl_display = ttl_seconds if ttl_seconds is not None else 0.0
    ttl_display_str = f"{ttl_display:.0f}s"

    last_updated = cache_stats.get("last_updated")
    last_updated_str = ""
    if isinstance(last_updated, str) and last_updated.strip() and last_updated != "-":
        last_updated_str = f" · Última actualización: {last_updated.strip()}"

    with st.container(border=True):
        state_map = {"green": "complete", "yellow": "running", "red": "error"}
        st.status(
            f"Cache: {ratio * 100:.1f}% hits · TTL restante: {ttl_display_str}"
            f"{last_updated_str}",
            state=state_map.get(color, "running"),
        )
    return color
