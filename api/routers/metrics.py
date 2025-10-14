"""Prometheus metrics exposure for the FastAPI backend."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

try:  # pragma: no cover - dependency guaranteed in normal installs
    from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest
except ModuleNotFoundError:  # pragma: no cover - fallback for missing optional dependency
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    generate_latest = None  # type: ignore[assignment]
    Gauge = None  # type: ignore[assignment]

from services import performance_timer as perf_timer

router = APIRouter()

if (
    Gauge is not None
    and getattr(perf_timer, "UI_TOTAL_LOAD_MS", None) is None
    and getattr(perf_timer, "PROMETHEUS_REGISTRY", None) is not None
):
    perf_timer.UI_TOTAL_LOAD_MS = Gauge(
        "ui_total_load_ms",
        "Total UI load time in milliseconds (from Streamlit startup to full render).",
        registry=perf_timer.PROMETHEUS_REGISTRY,
    )


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Expose accumulated Prometheus metrics for scraping."""

    registry = getattr(perf_timer, "PROMETHEUS_REGISTRY", None)
    enabled = bool(getattr(perf_timer, "PROMETHEUS_ENABLED", False))
    if not enabled or registry is None or generate_latest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prometheus metrics are disabled",
        )
    payload = generate_latest(registry)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


__all__ = ["router"]
