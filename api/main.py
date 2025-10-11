"""FastAPI application entrypoint."""

import logging

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from services.auth import AuthTokenError, verify_token
from shared.version import __version__

from .routers import cache, predictive, profile

logger = logging.getLogger(__name__)
logger.info("Starting FastAPI backend - version %s", __version__)

app = FastAPI(title="Portafolio IOL API", version=__version__)

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    """Validate bearer tokens and expose the associated claims."""

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token.",
        )
    try:
        return verify_token(credentials.credentials)
    except AuthTokenError as exc:  # pragma: no cover - exercised via tests
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc


app.include_router(predictive.router, dependencies=[Depends(get_current_user)])
app.include_router(profile.router)
app.include_router(cache.router, dependencies=[Depends(get_current_user)])


@app.post(
    "/predict",
    response_model=predictive.PredictResponse,
    summary="Predict sector performance (alias)",
    tags=["predictive"],
)
async def predict_alias(
    payload: predictive.PredictRequest,
    _claims: dict = Depends(get_current_user),
) -> predictive.PredictResponse:
    """Expose the predictive endpoint at the root level for convenience."""

    return await predictive.predict_sector(payload)


@app.post(
    "/forecast/adaptive",
    response_model=predictive.AdaptiveForecastResponse,
    summary="Simulate adaptive forecast (alias)",
    tags=["predictive"],
)
async def forecast_adaptive_alias(
    payload: predictive.AdaptiveForecastRequest,
    _claims: dict = Depends(get_current_user),
) -> predictive.AdaptiveForecastResponse:
    """Expose the adaptive forecast endpoint without the router prefix."""

    return await predictive.forecast_adaptive(payload)


@app.get("/health", summary="Service health status")
async def health() -> dict[str, str]:
    """Simple health-check endpoint for the API."""
    return {"status": "ok"}
