"""Cache management endpoints."""

from __future__ import annotations

from typing import Any, Mapping

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.cache.core import get_cache_stats

router = APIRouter(prefix="/cache", tags=["cache"])


try:  # pragma: no cover - support both Pydantic branches
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover
    ConfigDict = None  # type: ignore[assignment]


class _BaseModel(BaseModel):
    if ConfigDict is not None:  # pragma: no branch
        model_config = ConfigDict(extra="ignore")  # type: ignore[assignment]
    else:  # pragma: no cover - fallback for Pydantic v1
        class Config:  # type: ignore[override]
            extra = "ignore"


class CacheStatusResponse(_BaseModel):
    """Standardised cache stats payload."""

    namespace: str = ""
    hits: int = 0
    misses: int = 0
    hit_ratio: float = 0.0
    last_updated: str = "-"
    ttl_seconds: float | None = Field(default=None, ge=0.0)
    remaining_ttl: float | None = Field(default=None, ge=0.0)


@router.get("/", summary="Cache service placeholder")
async def cache_root() -> dict[str, str]:
    """Placeholder endpoint for cache services."""
    return {"detail": "Cache endpoints coming soon."}


@router.get("/status", response_model=CacheStatusResponse, summary="Cache statistics")
async def cache_status() -> CacheStatusResponse:
    """Expose cache statistics gathered from the core cache service."""

    stats = get_cache_stats()
    if not isinstance(stats, Mapping):
        stats = {}
    payload: dict[str, Any] = {
        "namespace": str(stats.get("namespace", "") or ""),
        "hits": int(stats.get("hits", 0) or 0),
        "misses": int(stats.get("misses", 0) or 0),
        "hit_ratio": float(stats.get("hit_ratio", 0.0) or 0.0),
        "last_updated": str(stats.get("last_updated", "-") or "-"),
        "ttl_seconds": stats.get("ttl_seconds"),
        "remaining_ttl": stats.get("remaining_ttl"),
    }
    return CacheStatusResponse(**payload)
