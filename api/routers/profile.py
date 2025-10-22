"""User profile endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import Field

from api.routers.base_models import _BaseModel
from services.auth import get_current_user

router = APIRouter(prefix="/profile", tags=["profile"], dependencies=[Depends(get_current_user)])


class ProfileSummaryResponse(_BaseModel):
    """Placeholder profile summary payload."""

    user_name: str = Field(..., description="Display name of the user")
    portfolio_value: float = Field(..., description="Total estimated value of the portfolio")
    currency: str = Field("ARS", description="Portfolio currency code")
    last_sync: datetime = Field(..., description="Timestamp of the last profile sync")


@router.get("/", summary="Profile service placeholder")
async def profile_root() -> dict[str, str]:
    """Placeholder endpoint for profile services."""
    return {"detail": "Profile endpoints coming soon."}


@router.get("/summary", response_model=ProfileSummaryResponse, summary="User profile summary")
async def profile_summary() -> ProfileSummaryResponse:
    """Expose a placeholder profile summary for the authenticated user."""

    now = datetime.utcnow()
    return ProfileSummaryResponse(
        user_name="Inversor Demo",
        portfolio_value=0.0,
        currency="ARS",
        last_sync=now,
    )
