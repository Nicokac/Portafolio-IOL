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
    perf_timer.UI_STARTUP_LOAD_MS = Gauge(
        "ui_startup_load_ms",
        "Login render latency in milliseconds before the authenticated UI loads.",
        registry=perf_timer.PROMETHEUS_REGISTRY,
    )
    perf_timer.PRELOAD_TOTAL_MS = Gauge(
        "preload_total_ms",
        "Total time spent importing scientific libraries after authentication in milliseconds.",
        registry=perf_timer.PROMETHEUS_REGISTRY,
    )
    perf_timer.PRELOAD_PANDAS_MS = Gauge(
        "preload_pandas_ms",
        "Duration of pandas preload in milliseconds.",
        registry=perf_timer.PROMETHEUS_REGISTRY,
    )
    perf_timer.PRELOAD_PLOTLY_MS = Gauge(
        "preload_plotly_ms",
        "Duration of plotly preload in milliseconds.",
        registry=perf_timer.PROMETHEUS_REGISTRY,
    )
    perf_timer.PRELOAD_STATSMODELS_MS = Gauge(
        "preload_statsmodels_ms",
        "Duration of statsmodels preload in milliseconds.",
        registry=perf_timer.PROMETHEUS_REGISTRY,
    )
    perf_timer._PRELOAD_LIBRARY_GAUGES.update(  # type: ignore[attr-defined]
        {
            "pandas": perf_timer.PRELOAD_PANDAS_MS,
            "plotly": perf_timer.PRELOAD_PLOTLY_MS,
            "statsmodels": perf_timer.PRELOAD_STATSMODELS_MS,
        }
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
