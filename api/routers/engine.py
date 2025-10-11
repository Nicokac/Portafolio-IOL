"""Engine service endpoints for predictive infrastructure."""

from __future__ import annotations

from datetime import datetime, timezone
import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)
logger.info("Initialising engine router")

router = APIRouter(prefix="/engine", tags=["engine"])


@router.get("/info", summary="Engine service metadata")
async def engine_info() -> dict[str, str]:
    """Return static metadata describing the predictive engine."""

    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "status": "ok",
        "engine_version": "v0.6.3",
        "timestamp": timestamp,
    }
