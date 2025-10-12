"""Cache management endpoints."""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from services.auth import get_current_user
from services.cache.market_data_cache import MarketDataCache, get_market_data_cache
from shared.errors import CacheUnavailableError, TimeoutError

LOGGER = logging.getLogger(__name__)

router = APIRouter(
    prefix="/cache",
    tags=["cache"],
    dependencies=[Depends(get_current_user)],
)


try:  # pragma: no cover - support both Pydantic branches
    from pydantic import ConfigDict, model_validator
    root_validator = None
except ImportError:  # pragma: no cover
    ConfigDict = None  # type: ignore[assignment]
    model_validator = None  # type: ignore[assignment]
    from pydantic import root_validator


class _BaseModel(BaseModel):
    if ConfigDict is not None:  # pragma: no branch
        model_config = ConfigDict(extra="ignore")  # type: ignore[assignment]
    else:  # pragma: no cover - fallback for Pydantic v1
        class Config:  # type: ignore[override]
            extra = "ignore"


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

        if model_validator is not None:  # pragma: no branch

            @model_validator(mode="after")
            def _check_payload(cls, values: "CacheInvalidateRequest") -> "CacheInvalidateRequest":
                if not values.pattern and not values.keys:
                    raise ValueError("Debe especificarse un 'pattern' o una lista de 'keys'")
                return values
    else:  # pragma: no cover - fallback for Pydantic v1
        class Config:  # type: ignore[override]
            extra = "ignore"

        if root_validator is not None:

            @root_validator(pre=True)
            def _check_payload(cls, values: Mapping[str, Any]) -> Mapping[str, Any]:
                pattern = values.get("pattern")
                keys = values.get("keys")
                if not pattern and not keys:
                    raise ValueError("Debe especificarse un 'pattern' o una lista de 'keys'")
                return values


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
async def cache_status() -> CacheStatusResponse:
    """Expose cache statistics gathered from the market data cache service."""

    try:
        cache = _market_cache()
        stats = cache.get_status_summary()
    except CacheUnavailableError as exc:
        LOGGER.exception("Cache status requested but unavailable")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="El servicio de caché no está disponible",
        ) from exc
    except TimeoutError as exc:
        LOGGER.exception("Cache status request timed out")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Timeout consultando el estado del caché",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Unexpected error while collecting cache status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error inesperado consultando el caché",
        ) from exc

    LOGGER.info("Cache status requested")
    return CacheStatusResponse(**stats)


@router.post("/invalidate", response_model=CacheInvalidateResponse, summary="Invalidate cache entries")
async def cache_invalidate(payload: CacheInvalidateRequest) -> CacheInvalidateResponse:
    """Invalidate cache entries based on an explicit list of keys or a glob pattern."""

    started = time.perf_counter()
    try:
        cache = _market_cache()
        removed = cache.invalidate_matching(keys=payload.keys, pattern=payload.pattern)
    except CacheUnavailableError as exc:
        LOGGER.exception("Cache invalidation failed due to unavailable backend")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="El servicio de caché no está disponible",
        ) from exc
    except TimeoutError as exc:
        LOGGER.exception("Cache invalidation timed out")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Timeout invalidando la caché",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Unexpected error during cache invalidation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error inesperado invalidando la caché",
        ) from exc

    elapsed = time.perf_counter() - started
    LOGGER.info("Invalidate executed", extra={"removed": removed})
    return CacheInvalidateResponse(removed=removed, elapsed_s=elapsed)


@router.post("/cleanup", response_model=CacheCleanupResponse, summary="Cleanup cache entries")
async def cache_cleanup() -> CacheCleanupResponse:
    """Remove expired or inconsistent records from the market data cache."""

    started = time.perf_counter()
    try:
        cache = _market_cache()
        metrics = cache.cleanup_expired()
    except CacheUnavailableError as exc:
        LOGGER.exception("Cache cleanup failed due to unavailable backend")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="El servicio de caché no está disponible",
        ) from exc
    except TimeoutError as exc:
        LOGGER.exception("Cache cleanup timed out")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Timeout limpiando la caché",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Unexpected error during cache cleanup")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error inesperado durante la limpieza de la caché",
        ) from exc

    elapsed = time.perf_counter() - started
    LOGGER.info(
        "Cleanup completed",
        extra={
            "expired_removed": metrics.get("expired_removed", 0),
            "orphans_removed": metrics.get("orphans_removed", 0),
        },
    )
    metrics_payload = dict(metrics)
    metrics_payload["elapsed_s"] = elapsed
    return CacheCleanupResponse(**metrics_payload)
