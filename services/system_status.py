from __future__ import annotations

"""Helpers to consolidate observability metrics for the system status panel."""

from dataclasses import dataclass
import time
from typing import Any, Mapping

import streamlit as st

from shared.time_provider import TimeProvider

_PROMETHEUS_SESSION_KEY = "prometheus_metrics"
_TOKEN_CLAIMS_KEY = "auth_token_claims"
_TOKEN_REFRESH_TS_KEY = "auth_token_refreshed_at"


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _flatten_metrics(target: dict[str, float], prefix: str, payload: Any) -> None:
    if isinstance(payload, Mapping):
        numeric_keys = ("value", "total", "count")
        for key in numeric_keys:
            numeric = _safe_float(payload.get(key))
            if numeric is not None:
                target[prefix] = numeric
                break
        else:
            for key, value in payload.items():
                nested_key = str(key or "").strip()
                if not nested_key:
                    continue
                composite = f"{prefix}_{nested_key}" if prefix else nested_key
                _flatten_metrics(target, composite, value)
        return
    numeric = _safe_float(payload)
    if numeric is not None:
        target[prefix] = numeric


def _parse_prometheus_source(source: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if isinstance(source, str):
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            value = _safe_float(parts[1])
            if value is None:
                continue
            metrics[parts[0]] = value
        return metrics
    if isinstance(source, Mapping):
        for key, value in source.items():
            name = str(key or "").strip()
            if not name:
                continue
            _flatten_metrics(metrics, name, value)
    return metrics


def _resolve_uptime(metrics: Mapping[str, float]) -> float | None:
    direct = metrics.get("uptime_seconds") or metrics.get("process_uptime_seconds")
    if direct is not None:
        return max(0.0, float(direct))
    start = metrics.get("process_start_time_seconds")
    if start is None:
        return None
    now = TimeProvider.now_datetime().timestamp()
    return max(0.0, now - float(start))


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class PrometheusSnapshot:
    """Aggregated metrics extracted from a Prometheus scrape."""

    uptime_seconds: float | None
    cache_hit_ratio: float | None
    auth_refresh_total: float | None
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class TokenSnapshot:
    """State associated with the currently issued authentication token."""

    active: bool
    username: str | None
    issued_at: int | None
    expires_at: int | None
    remaining_ttl: float | None
    ttl: int | None
    issued_label: str | None
    expires_label: str | None
    refreshed_at: str | None


@dataclass(frozen=True)
class SystemStatusSnapshot:
    """Container grouping Prometheus metrics and token state."""

    prometheus: PrometheusSnapshot
    token: TokenSnapshot


def get_prometheus_snapshot(source: Any | None = None) -> PrometheusSnapshot:
    """Return a normalised snapshot of Prometheus metrics."""

    if source is None:
        source = st.session_state.get(_PROMETHEUS_SESSION_KEY)
    metrics = _parse_prometheus_source(source)
    uptime_seconds = _resolve_uptime(metrics)
    cache_hit_ratio = metrics.get("cache_hit_ratio") or metrics.get("cache_hits_ratio")
    auth_refresh_total = metrics.get("auth_refresh_total") or metrics.get("auth_refresh_count")
    return PrometheusSnapshot(
        uptime_seconds=uptime_seconds,
        cache_hit_ratio=cache_hit_ratio,
        auth_refresh_total=auth_refresh_total,
        metrics=metrics,
    )


def get_token_snapshot(claims: Mapping[str, Any] | None = None) -> TokenSnapshot:
    """Return a snapshot of the active token using stored claims."""

    if claims is None:
        raw_claims = st.session_state.get(_TOKEN_CLAIMS_KEY)
        claims = raw_claims if isinstance(raw_claims, Mapping) else {}

    issued_at = _safe_int(claims.get("iat"))
    expires_at = _safe_int(claims.get("exp"))
    ttl = _safe_int(claims.get("ttl"))
    if ttl is None and issued_at is not None and expires_at is not None:
        ttl = max(0, expires_at - issued_at)
    username = claims.get("sub")
    if isinstance(username, str):
        username = username or None
    else:
        username = None

    remaining_ttl: float | None = None
    if expires_at is not None:
        remaining_ttl = max(0.0, float(expires_at - time.time()))

    issued_snapshot = TimeProvider.from_timestamp(issued_at)
    expires_snapshot = TimeProvider.from_timestamp(expires_at)
    refreshed_at = st.session_state.get(_TOKEN_REFRESH_TS_KEY)
    if isinstance(refreshed_at, str) and refreshed_at.strip():
        refreshed_label = refreshed_at.strip()
    else:
        refreshed_label = None

    return TokenSnapshot(
        active=bool(claims),
        username=username,
        issued_at=issued_at,
        expires_at=expires_at,
        remaining_ttl=remaining_ttl,
        ttl=ttl,
        issued_label=str(issued_snapshot) if issued_snapshot else None,
        expires_label=str(expires_snapshot) if expires_snapshot else None,
        refreshed_at=refreshed_label,
    )


def get_system_status_snapshot() -> SystemStatusSnapshot:
    """Collect the system status information to render in the UI."""

    prometheus = get_prometheus_snapshot()
    token = get_token_snapshot()
    return SystemStatusSnapshot(prometheus=prometheus, token=token)


__all__ = [
    "PrometheusSnapshot",
    "TokenSnapshot",
    "SystemStatusSnapshot",
    "get_prometheus_snapshot",
    "get_token_snapshot",
    "get_system_status_snapshot",
]

