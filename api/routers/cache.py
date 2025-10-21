"""Cache management endpoints."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Mapping

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import Field

from services import performance_timer as perf_timer
from services.auth import get_current_user
from services.cache.market_data_cache import MarketDataCache, get_market_data_cache
from shared.errors import CacheUnavailableError, TimeoutError
from api.routers.base_models import _BaseModel

try:  # pragma: no cover - optional dependency
    from prometheus_client import Counter, Summary
except ModuleNotFoundError:  # pragma: no cover - exercised when dependency missing
    Counter = None  # type: ignore[assignment]
    Summary = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)
PERF_LOGGER = getattr(perf_timer, "LOGGER", logging.getLogger("performance"))

router = APIRouter(
    prefix="/cache",
    tags=["cache"],
)

MAX_INVALIDATE_KEYS = 500


def _get_metric(name: str, factory: Any, *args: Any, **kwargs: Any) -> Any:
    registry = getattr(perf_timer, "PROMETHEUS_REGISTRY", None)
    enabled = bool(getattr(perf_timer, "PROMETHEUS_ENABLED", False))
    if not enabled or registry is None or factory is None:
        return None
    kwargs.setdefault("registry", registry)
    try:
        return factory(name, *args, **kwargs)
    except ValueError:
        existing = getattr(registry, "_names_to_collectors", {}).get(name)
        return existing


_CACHE_STATUS_COUNTER = _get_metric(
    "cache_status_requests_total",
    Counter,
    "Total de consultas al estado del caché",
    labelnames=("result",),
)
_CACHE_INVALIDATE_COUNTER = _get_metric(
    "cache_invalidate_total",
    Counter,
    "Total de invalidaciones de caché",
    labelnames=("result",),
)
_CACHE_CLEANUP_COUNTER = _get_metric(
    "cache_cleanup_total",
    Counter,
    "Total de limpiezas del caché",
    labelnames=("result",),
)
_CACHE_OPERATION_DURATION = _get_metric(
    "cache_operation_duration_seconds",
    Summary,
    "Duración de operaciones del caché",
    labelnames=("operation", "result"),
)

def _ensure_metrics() -> None:
    global _CACHE_STATUS_COUNTER, _CACHE_INVALIDATE_COUNTER, _CACHE_CLEANUP_COUNTER, _CACHE_OPERATION_DURATION
    if _CACHE_STATUS_COUNTER is None:
        _CACHE_STATUS_COUNTER = _get_metric(
            "cache_status_requests_total",
            Counter,
            "Total de consultas al estado del caché",
            labelnames=("result",),
        )
    if _CACHE_INVALIDATE_COUNTER is None:
        _CACHE_INVALIDATE_COUNTER = _get_metric(
            "cache_invalidate_total",
            Counter,
            "Total de invalidaciones de caché",
            labelnames=("result",),
        )
    if _CACHE_CLEANUP_COUNTER is None:
        _CACHE_CLEANUP_COUNTER = _get_metric(
            "cache_cleanup_total",
            Counter,
            "Total de limpiezas del caché",
            labelnames=("result",),
        )
    if _CACHE_OPERATION_DURATION is None:
        _CACHE_OPERATION_DURATION = _get_metric(
            "cache_operation_duration_seconds",
            Summary,
            "Duración de operaciones del caché",
            labelnames=("operation", "result"),
        )


def _resolve_username(user: Mapping[str, Any] | None) -> str:
    if not isinstance(user, Mapping):
        return "anonymous"
    username = user.get("sub") or user.get("username") or user.get("user")
    if isinstance(username, str) and username.strip():
        return username
    return "anonymous"


def _record_metrics(operation: str, success: bool, elapsed: float) -> None:
    _ensure_metrics()
    result = "success" if success else "error"
    counter_map = {
        "status": _CACHE_STATUS_COUNTER,
        "invalidate": _CACHE_INVALIDATE_COUNTER,
        "cleanup": _CACHE_CLEANUP_COUNTER,
    }
    counter = counter_map.get(operation)
    if counter is not None:
        try:
            counter.labels(result=result).inc()
        except Exception:  # pragma: no cover - defensive guard against registry issues
            pass
    if _CACHE_OPERATION_DURATION is not None:
        try:
            _CACHE_OPERATION_DURATION.labels(
                operation=operation, result=result
            ).observe(elapsed)
        except Exception:  # pragma: no cover - defensive guard against registry issues
            pass


def _emit_structured_log(
    *,
    operation: str,
    user: Mapping[str, Any] | None,
    entries: int,
    elapsed: float,
    success: bool,
) -> None:
    payload = {
        "operation": operation,
        "user": _resolve_username(user),
        "entries": int(entries),
        "elapsed_s": float(elapsed),
        "success": bool(success),
    }
    try:
        PERF_LOGGER.info(json.dumps(payload, ensure_ascii=False))
    except Exception:  # pragma: no cover - logging must never raise
        LOGGER.debug("Failed to emit structured cache log", exc_info=True)


def _finalize_operation(
    *,
    operation: str,
    user: Mapping[str, Any] | None,
    entries: int,
    started: float,
    success: bool,
) -> float:
    elapsed = time.perf_counter() - started
    _record_metrics(operation, success, elapsed)
    _emit_structured_log(
        operation=operation,
        user=user,
        entries=entries,
        elapsed=elapsed,
        success=success,
    )
    return elapsed


def _validate_invalidate_targets(
    payload: CacheInvalidateRequest,
) -> tuple[list[str] | None, str | None]:
    pattern = payload.pattern
    normalized_pattern = pattern.strip() if isinstance(pattern, str) else pattern
    if normalized_pattern == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid pattern",
        )

    keys = payload.keys
    normalized_keys: list[str] | None = None
    if keys is not None:
        normalized_keys = [key for key in keys if isinstance(key, str) and key.strip()]
        if not normalized_keys and normalized_pattern is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="invalid pattern",
            )
        if normalized_keys and len(normalized_keys) > MAX_INVALIDATE_KEYS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"max keys exceeded ({MAX_INVALIDATE_KEYS})",
            )
        if normalized_keys == []:
            normalized_keys = None

    if normalized_pattern is None and normalized_keys is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid pattern",
        )

    return normalized_keys, normalized_pattern


try:  # pragma: no cover - support both Pydantic branches
    from pydantic import ConfigDict, model_validator
    root_validator = None
except ImportError:  # pragma: no cover
    ConfigDict = None  # type: ignore[assignment]
    model_validator = None  # type: ignore[assignment]
    from pydantic import root_validator


class CacheStatusResponse(_BaseModel):
    """Resumen de métricas del caché de mercado."""

    total_entries: int = Field(0, ge=0, description="Cantidad total de entradas almacenadas")
    hit_ratio: float = Field(0.0, ge=0.0, description="Porcentaje de aciertos acumulados")
    avg_ttl_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="TTL promedio (en segundos) para las entradas con expiración",
    )
    size_mb: float = Field(0.0, ge=0.0, description="Tamaño estimado del caché en MB")


class CacheInvalidateRequest(_BaseModel):
    """Payload para invalidar entradas de caché."""

    pattern: str | None = Field(
        default=None,
        description="Patrón glob para invalidar claves (por ejemplo symbol_*)",
    )
    keys: list[str] | None = Field(
        default=None,
        description="Listado explícito de claves a invalidar",
    )

    if ConfigDict is not None:  # pragma: no branch
        model_config = ConfigDict(extra="ignore")  # type: ignore[assignment]

    else:  # pragma: no cover - fallback for Pydantic v1
        class Config:  # type: ignore[override]
            extra = "ignore"



class CacheInvalidateResponse(_BaseModel):
    """Resultado de la operación de invalidación."""

    removed: int = Field(..., ge=0, description="Cantidad de entradas eliminadas")
    elapsed_s: float = Field(..., ge=0.0, description="Tiempo total de la operación en segundos")


class CacheCleanupResponse(_BaseModel):
    """Resultado de la limpieza del caché."""

    expired_removed: int = Field(..., ge=0, description="Registros expirados eliminados")
    orphans_removed: int = Field(..., ge=0, description="Registros inconsistentes eliminados")
    elapsed_s: float = Field(..., ge=0.0, description="Duración de la limpieza en segundos")


def _market_cache() -> MarketDataCache:
    cache = get_market_data_cache()
    if cache is None:
        raise CacheUnavailableError("Servicio de caché no disponible")
    return cache


@router.get("/status", response_model=CacheStatusResponse, summary="Cache statistics")
async def cache_status(user: Mapping[str, Any] = Depends(get_current_user)) -> CacheStatusResponse:
    """Expose cache statistics gathered from the market data cache service."""

    started = time.perf_counter()
    entries = 0
    try:
        cache = _market_cache()
        stats = cache.get_status_summary()
        entries = int(stats.get("total_entries", 0) or 0)
    except CacheUnavailableError as exc:
        LOGGER.exception("Cache status requested but unavailable")
        _finalize_operation(
            operation="status",
            user=user,
            entries=entries,
            started=started,
            success=False,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="El servicio de caché no está disponible",
        ) from exc
    except TimeoutError as exc:
        LOGGER.exception("Cache status request timed out")
        _finalize_operation(
            operation="status",
            user=user,
            entries=entries,
            started=started,
            success=False,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Timeout consultando el estado del caché",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Unexpected error while collecting cache status")
        _finalize_operation(
            operation="status",
            user=user,
            entries=entries,
            started=started,
            success=False,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error inesperado consultando el caché",
        ) from exc

    _finalize_operation(
        operation="status",
        user=user,
        entries=entries,
        started=started,
        success=True,
    )
    return CacheStatusResponse(**stats)


@router.post("/invalidate", response_model=CacheInvalidateResponse, summary="Invalidate cache entries")
async def cache_invalidate(
    payload: CacheInvalidateRequest,
    user: Mapping[str, Any] = Depends(get_current_user),
) -> CacheInvalidateResponse:
    """Invalidate cache entries based on an explicit list of keys or a glob pattern."""

    started = time.perf_counter()
    removed = 0
    try:
        keys, pattern = _validate_invalidate_targets(payload)
    except HTTPException as exc:
        _finalize_operation(
            operation="invalidate",
            user=user,
            entries=0,
            started=started,
            success=False,
        )
        raise exc
    try:
        cache = _market_cache()
        removed = cache.invalidate_matching(keys=keys, pattern=pattern)
    except CacheUnavailableError as exc:
        LOGGER.exception("Cache invalidation failed due to unavailable backend")
        _finalize_operation(
            operation="invalidate",
            user=user,
            entries=removed,
            started=started,
            success=False,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="El servicio de caché no está disponible",
        ) from exc
    except TimeoutError as exc:
        LOGGER.exception("Cache invalidation timed out")
        _finalize_operation(
            operation="invalidate",
            user=user,
            entries=removed,
            started=started,
            success=False,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Timeout invalidando la caché",
        ) from exc
    except ValueError as exc:
        _finalize_operation(
            operation="invalidate",
            user=user,
            entries=removed,
            started=started,
            success=False,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Unexpected error during cache invalidation")
        _finalize_operation(
            operation="invalidate",
            user=user,
            entries=removed,
            started=started,
            success=False,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error inesperado invalidando la caché",
        ) from exc

    elapsed = _finalize_operation(
        operation="invalidate",
        user=user,
        entries=removed,
        started=started,
        success=True,
    )
    return CacheInvalidateResponse(removed=removed, elapsed_s=elapsed)


@router.post("/cleanup", response_model=CacheCleanupResponse, summary="Cleanup cache entries")
async def cache_cleanup(user: Mapping[str, Any] = Depends(get_current_user)) -> CacheCleanupResponse:
    """Remove expired or inconsistent records from the market data cache."""

    started = time.perf_counter()
    try:
        cache = _market_cache()
        metrics = cache.cleanup_expired()
    except CacheUnavailableError as exc:
        LOGGER.exception("Cache cleanup failed due to unavailable backend")
        _finalize_operation(
            operation="cleanup",
            user=user,
            entries=0,
            started=started,
            success=False,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="El servicio de caché no está disponible",
        ) from exc
    except TimeoutError as exc:
        LOGGER.exception("Cache cleanup timed out")
        _finalize_operation(
            operation="cleanup",
            user=user,
            entries=0,
            started=started,
            success=False,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Timeout limpiando la caché",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Unexpected error during cache cleanup")
        _finalize_operation(
            operation="cleanup",
            user=user,
            entries=0,
            started=started,
            success=False,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error inesperado durante la limpieza de la caché",
        ) from exc

    total_entries = int(metrics.get("expired_removed", 0) or 0) + int(
        metrics.get("orphans_removed", 0) or 0
    )
    elapsed = _finalize_operation(
        operation="cleanup",
        user=user,
        entries=total_entries,
        started=started,
        success=True,
    )
    metrics_payload = dict(metrics)
    metrics_payload["elapsed_s"] = elapsed
    return CacheCleanupResponse(**metrics_payload)
