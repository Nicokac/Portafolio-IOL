"""Predictive analytics endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/predictive", tags=["predictive"])


@router.get("/", summary="Predictive service placeholder")
async def predictive_root() -> dict[str, str]:
    """Placeholder endpoint for predictive services."""
    return {"detail": "Predictive endpoints coming soon."}
