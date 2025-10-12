"""FastAPI application entrypoint."""

import logging

from fastapi import Depends, FastAPI

from services.auth import get_current_user
from shared.security_env_validator import validate_security_environment
from shared.version import __version__

from .routers import auth, cache, engine, metrics, predictive, profile

logger = logging.getLogger(__name__)
logger.info("Starting FastAPI backend - version %s", __version__)

validate_security_environment()

app = FastAPI(title="Portafolio IOL API", version=__version__)


app.include_router(predictive.router, dependencies=[Depends(get_current_user)])
app.include_router(cache.router)
app.include_router(profile.router)
app.include_router(auth.router)
app.include_router(engine.router)
app.include_router(metrics.router)


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
