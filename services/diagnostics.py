from __future__ import annotations

"""Utilities for collecting and reporting startup diagnostics."""

import logging
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

import streamlit as st

from services import snapshots
from services.environment import capture_environment_snapshot
from services.health import get_health_metrics, record_environment_snapshot
from shared.cache import cache as shared_cache
from shared.export import ensure_kaleido_runtime
from shared.time_provider import TimeProvider


analysis_logger = logging.getLogger("analysis")
logger = logging.getLogger(__name__)


_IOL_HOST = "api.invertironline.com"
_IOL_PORT = 443


_STATUS_ICONS: Mapping[str, str] = {
    "success": "✅",
    "ok": "✅",
    "warning": "⚠️",
    "degraded": "⚠️",
    "error": "❌",
    "critical": "❌",
    "stale": "🟡",
    "info": "ℹ️",
}


def _normalize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(val) for key, val in value.items()}
    return str(value)


_SENSITIVE_SESSION_KEYS = {
    "auth_token",
    "authorization",
    "bearer_token",
    "fastapi_tokens_key",
    "fernet_key",
    "iol_password",
    "iol_tokens_key",
    "iol_username",
    "password",
    "refresh_token",
    "secret",
    "token",
    "tokens_file",
    "username",
}

_SENSITIVE_KEY_HINTS = (
    "password",
    "secret",
    "token",
    "key",
    "username",
    "credential",
)


def _is_sensitive_key(key: str) -> bool:
    normalized = str(key or "").strip().casefold()
    if not normalized:
        return False
    if normalized in _SENSITIVE_SESSION_KEYS:
        return True
    return any(hint in normalized for hint in _SENSITIVE_KEY_HINTS)


def _sanitize_session_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for nested_key, nested_value in value.items():
            if _is_sensitive_key(str(nested_key)):
                sanitized[str(nested_key)] = "[REDACTED]"
            else:
                sanitized[str(nested_key)] = _sanitize_session_value(nested_value)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_session_value(item) for item in value]
    return _normalize_value(value)


def _snapshot_session_state(state: Mapping[str, Any]) -> dict[str, Any]:
    session_id = state.get("session_id")
    snapshot: dict[str, Any] = {
        "id": str(session_id) if session_id is not None else None,
    }

    values: dict[str, Any] = {}
    flags: list[str] = []

    for key in sorted(state):
        if key == "session_id" or key.startswith("_"):
            continue
        value = state[key]
        if _is_sensitive_key(key):
            values[key] = "[REDACTED]"
            continue
        if isinstance(value, bool):
            values[key] = value
            if value:
                flags.append(key)
            continue
        values[key] = _sanitize_session_value(value)

    if values:
        snapshot["values"] = values
    if flags:
        snapshot["flags"] = sorted(flags)

    return snapshot


def _coerce_icon(entry: Mapping[str, Any], *, fallback: str) -> str:
    icon = entry.get("icon")
    if isinstance(icon, str) and icon.strip():
        return icon.strip()

    status = entry.get("status")
    status_key = str(status or "").strip().casefold()
    mapped = _STATUS_ICONS.get(status_key)
    if mapped:
        return mapped
    return "ℹ️"


def _resolve_label(entry: Mapping[str, Any], *, fallback: str) -> str:
    label = entry.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    name = entry.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return fallback


def _resolve_value(entry: Mapping[str, Any]) -> str:
    status = entry.get("status")
    detail = entry.get("detail")
    value = None
    if isinstance(status, str) and status.strip():
        value = status.strip()
    raw_value = entry.get("value")
    if value is None and isinstance(raw_value, str) and raw_value.strip():
        value = raw_value.strip()
    if value is None:
        numeric = entry.get("elapsed_ms") or entry.get("latency_ms")
        if isinstance(numeric, (int, float)):
            value = f"{float(numeric):.0f} ms"
    if value is None:
        value = "s/d"

    if isinstance(detail, str) and detail.strip():
        return f"{value} — {detail.strip()}"
    return value


def _collect_highlights(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    highlights: list[dict[str, Any]] = []
    if not isinstance(metrics, Mapping):
        return highlights

    for key in sorted(metrics):
        entry = metrics.get(key)
        if not isinstance(entry, Mapping):
            continue
        label = _resolve_label(entry, fallback=key.replace("_", " ").title())
        icon = _coerce_icon(entry, fallback=label)
        value = _resolve_value(entry)
        highlights.append(
            {
                "id": key,
                "icon": icon,
                "label": label,
                "value": value,
            }
        )

    return highlights


@dataclass(frozen=True)
class DiagnosticCheck:
    """Represents the outcome of an individual diagnostic step."""

    component: str
    status: str
    message: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "component": self.component,
            "status": self.status,
            "message": self.message,
        }


def _check_iol_connectivity(timeout: float = 2.5) -> DiagnosticCheck:
    host = _IOL_HOST
    port = _IOL_PORT
    try:
        with socket.create_connection((host, port), timeout):
            pass
    except OSError as exc:  # pragma: no cover - depends on network conditions
        raise RuntimeError(f"Conectividad IOL falló: {exc}") from exc
    return DiagnosticCheck("iol_connectivity", "ok", "Conectividad con IOL verificada")


def _check_bearer_token(tokens: Mapping[str, Any] | None) -> DiagnosticCheck:
    if not isinstance(tokens, Mapping):
        raise RuntimeError("Tokens de autenticación no disponibles")

    access_token = tokens.get("access_token") or tokens.get("accessToken")
    token_type = tokens.get("token_type") or tokens.get("tokenType")
    if not access_token or not isinstance(access_token, str):
        raise RuntimeError("access_token ausente tras login")
    if not token_type or str(token_type).strip().lower() != "bearer":
        raise RuntimeError("token_type inválido o distinto de Bearer")

    expires_in = tokens.get("expires_in") or tokens.get("expiresIn")
    if expires_in is not None:
        try:
            expires_value = float(expires_in)
        except (TypeError, ValueError):
            raise RuntimeError("expires_in inválido en tokens de autenticación")
        if expires_value <= 0:
            raise RuntimeError("El bearer token reporta expiración inmediata")

    return DiagnosticCheck("bearer_token", "ok", "Bearer token válido")


def _check_snapshots_access() -> DiagnosticCheck:
    try:
        snapshots.auto_configure_if_needed()
        snapshots.list_snapshots(limit=1)
    except Exception as exc:
        raise RuntimeError(f"Backend de snapshots inaccesible: {exc}") from exc
    return DiagnosticCheck("snapshots", "ok", "Backend de snapshots disponible")


def _check_kaleido_runtime() -> DiagnosticCheck:
    try:
        available = ensure_kaleido_runtime()
    except Exception as exc:  # pragma: no cover - defensive, depends on runtime
        logger.debug("Error verificando runtime de Kaleido", exc_info=exc)
        available = False
    if not available:
        return DiagnosticCheck(
            "kaleido",
            "warning",
            "Kaleido no está disponible; las exportaciones PNG seguirán deshabilitadas",
        )
    return DiagnosticCheck("kaleido", "ok", "Runtime de Kaleido disponible")


def _check_cache_state() -> DiagnosticCheck:
    probe_key = "_diagnostics_cache_probe"
    sentinel = object()
    marker = {"ts": time.time()}
    previous = shared_cache.get(probe_key, sentinel)
    try:
        shared_cache.set(probe_key, marker)
        stored = shared_cache.get(probe_key)
        if stored != marker:
            raise RuntimeError("La caché no devolvió el valor esperado")
    finally:
        if previous is sentinel:
            shared_cache.pop(probe_key, None)
        else:
            shared_cache.set(probe_key, previous)
    return DiagnosticCheck("cache", "ok", "Cachés de sesión operativas")


def _run_runtime_checks(*, tokens: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    started_at = time.perf_counter()
    timestamp = time.time()

    checks: list[DiagnosticCheck] = []
    errors = 0
    warnings = 0

    def _run(checker: Callable[[], DiagnosticCheck]) -> None:
        nonlocal errors, warnings
        name = getattr(checker, "__name__", "check")
        try:
            result = checker()
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.warning("Diagnóstico %s falló: %s", name, exc)
            checks.append(
                DiagnosticCheck(name, "error", str(exc)),
            )
            errors += 1
            return
        checks.append(result)
        if result.status == "warning":
            warnings += 1
        elif result.status not in ("ok", "success"):
            errors += 1

    _run(lambda: _check_iol_connectivity())
    if tokens is not None:
        _run(lambda: _check_bearer_token(tokens))
    else:
        checks.append(
            DiagnosticCheck(
                "bearer_token",
                "warning",
                "Tokens de autenticación no disponibles",
            )
        )
        warnings += 1
    _run(_check_snapshots_access)
    _run(_check_kaleido_runtime)
    _run(_check_cache_state)

    overall_status = "ok"
    if errors:
        overall_status = "error"
    elif warnings:
        overall_status = "degraded"

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    summary = "; ".join(f"{check.component}: {check.message}" for check in checks)

    return {
        "status": overall_status,
        "latency": latency_ms,
        "timestamp": timestamp,
        "component": "startup_diagnostics",
        "message": summary,
        "checks": [check.as_dict() for check in checks],
    }


def run_startup_diagnostics(*, tokens: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Gather startup diagnostics combining telemetry and runtime checks."""

    metrics = get_health_metrics()
    session_snapshot = _snapshot_session_state(st.session_state)
    timestamp = TimeProvider.now()
    environment_snapshot = capture_environment_snapshot()

    payload: Dict[str, Any] = {
        "event": "startup.diagnostics",
        "timestamp": timestamp,
        "session": session_snapshot,
        "metrics": metrics,
        "highlights": _collect_highlights(metrics),
        "environment": environment_snapshot,
    }

    runtime_diagnostics = _run_runtime_checks(tokens=tokens)
    payload.update(
        {
            "component": runtime_diagnostics["component"],
            "status": runtime_diagnostics["status"],
            "latency": runtime_diagnostics["latency"],
            "checks": runtime_diagnostics["checks"],
            "runtime": runtime_diagnostics,
        }
    )

    try:
        record_environment_snapshot(environment_snapshot)
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.warning("No se pudo registrar snapshot de entorno", exc_info=exc)

    analysis_logger.info("startup.diagnostics", extra={"analysis": payload})
    return payload


__all__ = ["run_startup_diagnostics"]
