"""User profile endpoints."""

from fastapi import APIRouter

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("/", summary="Profile service placeholder")
async def profile_root() -> dict[str, str]:
    """Placeholder endpoint for profile services."""
    return {"detail": "Profile endpoints coming soon."}
