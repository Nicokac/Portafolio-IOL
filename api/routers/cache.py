"""Cache management endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/cache", tags=["cache"])


@router.get("/", summary="Cache service placeholder")
async def cache_root() -> dict[str, str]:
    """Placeholder endpoint for cache services."""
    return {"detail": "Cache endpoints coming soon."}
