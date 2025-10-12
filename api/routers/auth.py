"""Authentication endpoints for token management."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from services.auth import AuthTokenError, refresh_active_token, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


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


class RefreshTokenResponse(_BaseModel):
    """Standard response payload for refreshed tokens."""

    access_token: str = Field(..., description="Fresh access token")
    token_type: str = Field("bearer", description="Type of the authentication token")
    expires_in: int = Field(..., ge=1, description="Seconds until the token expires")
    issued_at: int = Field(..., ge=0, description="Unix timestamp when the token was issued")
    session_id: str = Field(..., description="Session identifier associated with the token")


@router.post("/refresh", response_model=RefreshTokenResponse, summary="Refresh authentication token")
async def refresh_token_endpoint(claims: dict = Depends(get_current_user)) -> RefreshTokenResponse:
    """Issue a new token when the current one is close to expiring."""

    try:
        result = refresh_active_token(claims)
    except AuthTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

    new_claims = result["claims"]
    return RefreshTokenResponse(
        access_token=result["token"],
        expires_in=int(new_claims["exp"]) - int(new_claims["iat"]),
        issued_at=int(new_claims["iat"]),
        session_id=str(new_claims["session_id"]),
    )


__all__ = ["router"]
