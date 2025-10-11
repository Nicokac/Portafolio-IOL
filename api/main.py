"""FastAPI application entrypoint."""

import logging

from fastapi import FastAPI

from shared.version import __version__

from .routers import cache, predictive, profile

logger = logging.getLogger(__name__)
logger.info("Starting FastAPI backend - version %s", __version__)

app = FastAPI(title="Portafolio IOL API", version=__version__)

app.include_router(predictive.router)
app.include_router(profile.router)
app.include_router(cache.router)


@app.get("/health", summary="Service health status")
async def health() -> dict[str, str]:
    """Simple health-check endpoint for the API."""
    return {"status": "ok"}
