from __future__ import annotations

"""Runtime diagnostics executed after successful IOL authentication."""

import logging
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

from services import snapshots
from shared.cache import cache as shared_cache
from shared.export import ensure_kaleido_runtime


logger = logging.getLogger(__name__)

_IOL_HOST = "api.invertironline.com"
_IOL_PORT = 443


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
    try:
        with socket.create_connection((host, _IOL_PORT), timeout):
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


def run_startup_diagnostics(*, tokens: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Execute lightweight checks to assess the environment health."""

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
    _run(lambda: _check_bearer_token(tokens))
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


__all__ = ["run_startup_diagnostics"]
