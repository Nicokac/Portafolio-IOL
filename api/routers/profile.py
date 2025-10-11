"""User profile endpoints."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/profile", tags=["profile"])


try:  # pragma: no cover - compatibility across Pydantic versions
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover
    ConfigDict = None  # type: ignore[assignment]


class _BaseModel(BaseModel):
    if ConfigDict is not None:  # pragma: no branch
        model_config = ConfigDict(extra="ignore")  # type: ignore[assignment]
    else:  # pragma: no cover - fallback for Pydantic v1
        class Config:  # type: ignore[override]
            extra = "ignore"


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
