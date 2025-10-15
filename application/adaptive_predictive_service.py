from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from application.backtesting_service import BacktestingService
from application.predictive_core import (
    PredictiveCacheState,
    run_backtest,
)
from application.predictive_service import build_adaptive_history
from services.cache import CacheService
from services.performance_metrics import track_function
from services.performance_timer import ProfileBlockResult, profile_block
from shared.settings import ADAPTIVE_TTL_HOURS

from predictive_engine import __version__ as ENGINE_VERSION
from predictive_engine.adapters import run_adaptive_forecast
from predictive_engine.models import AdaptiveUpdateResult, empty_history_frame
from domain.adaptive_cache_lock import adaptive_cache_lock

LOGGER = logging.getLogger(__name__)

_CACHE_NAMESPACE = "adaptive_predictive"
_STATE_KEY = "adaptive_state"
_CORR_KEY = "adaptive_correlations"
_MAX_HISTORY_ROWS = 720
_DEFAULT_EMA_SPAN = 5
_HISTORY_PATH = Path("./data/forecast_history.parquet")

_CACHE = CacheService(
    namespace=_CACHE_NAMESPACE,
    ttl_override=ADAPTIVE_TTL_HOURS * 3600.0,
)
_CACHE_STATE = PredictiveCacheState()

_LOCK_PROLONGED_THRESHOLD_S = 120.0


def _warn_prolonged_lock(profile: ProfileBlockResult | None, *, operation: str) -> None:
    if not isinstance(profile, ProfileBlockResult):
        return
    duration = float(profile.duration_s or 0.0)
    if duration <= _LOCK_PROLONGED_THRESHOLD_S:
        return
    LOGGER.warning(
        "Retención prolongada del lock adaptativo en %s: %.2fs",
        operation,
        duration,
    )


def _cache_last_updated(cache: CacheService) -> str | None:
    try:
        value = getattr(cache, "last_updated_human", "-")
    except Exception:  # pragma: no cover - defensive
        return None
    if not value or value == "-":
        return None
    return str(value)


def update_model(
    predictions: pd.DataFrame | None,
    actuals: pd.DataFrame | None,
    *,
    cache: CacheService | None = None,
    ema_span: int = _DEFAULT_EMA_SPAN,
    timestamp: pd.Timestamp | None = None,
    persist: bool = True,
    ttl_hours: float | None = None,
) -> dict[str, Any]:
    """Update the adaptive state using normalized prediction errors."""

    LOGGER.debug("Ejecutando predictive engine %s", ENGINE_VERSION)

    effective_ttl_hours = (
        float(ttl_hours)
        if ttl_hours is not None
        else float(ADAPTIVE_TTL_HOURS)
    )

    active_cache = cache or _CACHE
    ttl_seconds = max(effective_ttl_hours, 0.0) * 3600.0
    if isinstance(active_cache, CacheService):
        active_cache.set_ttl_override(ttl_seconds)

    LOGGER.debug(
        "Solicitando lock adaptativo para update_model (ema_span=%s, persist=%s)",
        ema_span,
        persist,
    )
    lock_profile: ProfileBlockResult | None = None
    with adaptive_cache_lock:
        LOGGER.debug(
            "Lock adaptativo adquirido para update_model (ema_span=%s)",
            ema_span,
        )
        with profile_block(
            "adaptive_predictive.lock_scope.update",
            extra={
                "operation": "update_model",
                "ema_span": str(ema_span),
            },
            module=__name__,
            threshold_s=_LOCK_PROLONGED_THRESHOLD_S,
        ) as lock_profile_ctx:
            lock_profile = lock_profile_ctx
            engine_result = run_adaptive_forecast(
                predictions=predictions,
                actuals=actuals,
                cache=active_cache,
                ema_span=ema_span,
                ttl_hours=effective_ttl_hours,
                max_history_rows=_MAX_HISTORY_ROWS,
                persist_state=persist,
                persist_history=persist,
                history_path=_HISTORY_PATH,
                warm_start=True,
                state_key=_STATE_KEY,
                correlation_key=_CORR_KEY,
                timestamp=timestamp,
                performance_prefix="predictive",
            )
    LOGGER.debug(
        "Lock adaptativo liberado tras update_model (ema_span=%s, persist=%s)",
        ema_span,
        persist,
    )
    _warn_prolonged_lock(lock_profile, operation="update_model")

    update_result = engine_result.get("update")
    cache_hit = bool(engine_result.get("cache_hit"))
    cache_metadata = engine_result.get("cache_metadata") or {}
    cache_timestamp = _cache_last_updated(active_cache) or cache_metadata.get("last_updated")

    if cache_hit:
        _CACHE_STATE.record_hit(
            last_updated=cache_timestamp,
            ttl_hours=effective_ttl_hours,
        )
    else:
        _CACHE_STATE.record_miss(
            last_updated=cache_timestamp,
            ttl_hours=effective_ttl_hours,
        )

    payload: dict[str, Any] = {}
    if isinstance(update_result, AdaptiveUpdateResult):
        payload = update_result.to_dict()
    if cache_metadata:
        payload["cache_metadata"] = cache_metadata
    return payload


def prepare_adaptive_history(
    opportunities: pd.DataFrame | None,
    *,
    backtesting_service: BacktestingService | None = None,
    span: int = _DEFAULT_EMA_SPAN,
    max_symbols: int = 12,
) -> pd.DataFrame:
    """Wrapper maintained for compatibility, delegates to build_adaptive_history."""

    return build_adaptive_history(
        opportunities,
        mode="real",
        backtesting_service=backtesting_service,
        span=span,
        max_symbols=max_symbols,
    )


def generate_synthetic_history(
    recommendations: pd.DataFrame, periods: int = 6
) -> pd.DataFrame:
    """Wrapper maintained for compatibility, delegates to build_adaptive_history."""

    return build_adaptive_history(
        recommendations,
        mode="synthetic",
        periods=periods,
    )


@track_function("simulate_adaptive_forecast")
def simulate_adaptive_forecast(
    history: pd.DataFrame | None,
    *,
    ema_span: int = _DEFAULT_EMA_SPAN,
    cache: CacheService | None = None,
    persist: bool = True,
    rolling_window: int = 20,
    ttl_hours: float | None = None,
) -> dict[str, Any]:
    """Run an adaptive backtest and expose error metrics and correlations."""

    LOGGER.debug("Ejecutando predictive engine %s", ENGINE_VERSION)

    effective_ttl_hours = (
        float(ttl_hours)
        if ttl_hours is not None
        else float(ADAPTIVE_TTL_HOURS)
    )
    ttl_seconds = max(effective_ttl_hours, 0.0) * 3600.0
    working_cache = cache or (
        _CACHE if persist else CacheService(namespace=f"{_CACHE_NAMESPACE}_sim")
    )
    if isinstance(working_cache, CacheService):
        working_cache.set_ttl_override(ttl_seconds)

    frame = pd.DataFrame()
    if isinstance(history, pd.DataFrame) and not history.empty:
        frame = history.copy()
        if "timestamp" not in frame.columns:
            frame["timestamp"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(frame))
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "sector", "predicted_return", "actual_return"])
        frame = frame.sort_values("timestamp")

    LOGGER.debug(
        "Solicitando lock adaptativo para simulate_adaptive_forecast (ema_span=%s, persist=%s)",
        ema_span,
        persist,
    )
    lock_profile: ProfileBlockResult | None = None
    with adaptive_cache_lock:
        LOGGER.debug(
            "Lock adaptativo adquirido para simulate_adaptive_forecast (ema_span=%s)",
            ema_span,
        )
        with profile_block(
            "adaptive_predictive.lock_scope.forecast",
            extra={
                "operation": "simulate_adaptive_forecast",
                "ema_span": str(ema_span),
                "persist": str(bool(persist)),
            },
            module=__name__,
            threshold_s=_LOCK_PROLONGED_THRESHOLD_S,
        ) as lock_profile_ctx:
            lock_profile = lock_profile_ctx
            engine_result = run_adaptive_forecast(
                history=frame,
                cache=working_cache,
                ema_span=ema_span,
                rolling_window=rolling_window,
                ttl_hours=effective_ttl_hours,
                max_history_rows=_MAX_HISTORY_ROWS,
                persist_state=persist,
                persist_history=persist,
                history_path=_HISTORY_PATH,
                warm_start=True,
                state_key=_STATE_KEY,
                correlation_key=_CORR_KEY,
                performance_prefix="predictive",
            )
    LOGGER.debug(
        "Lock adaptativo liberado tras simulate_adaptive_forecast (ema_span=%s, persist=%s)",
        ema_span,
        persist,
    )
    _warn_prolonged_lock(lock_profile, operation="simulate_adaptive_forecast")

    forecast_result = engine_result.get("forecast")
    cache_metadata = engine_result.get("cache_metadata") or {}
    cache_last_updated = cache_metadata.get("last_updated")
    if (not cache_last_updated or cache_last_updated == "-") and not frame.empty:
        last_timestamp = frame["timestamp"].max()
        if isinstance(last_timestamp, pd.Timestamp):
            cache_last_updated = last_timestamp.strftime("%H:%M:%S")
            cache_metadata["last_updated"] = cache_last_updated

    cache_hit = bool(engine_result.get("cache_hit"))
    cache_timestamp = _cache_last_updated(working_cache) or cache_last_updated
    if cache_hit:
        _CACHE_STATE.record_hit(last_updated=cache_timestamp, ttl_hours=effective_ttl_hours)
    else:
        _CACHE_STATE.record_miss(last_updated=cache_timestamp, ttl_hours=effective_ttl_hours)

    payload: dict[str, Any] = {}
    if hasattr(forecast_result, "as_dict"):
        payload = forecast_result.as_dict()  # type: ignore[call-arg]
    elif isinstance(forecast_result, dict):
        payload = dict(forecast_result)
    if cache_metadata:
        payload.setdefault("cache_metadata", {}).update(cache_metadata)
    beta_shift = payload.get("beta_shift", pd.Series(dtype=float))
    summary = payload.get("summary", {})
    if isinstance(beta_shift, pd.Series) and not beta_shift.empty:
        summary.setdefault("beta_mean", float(beta_shift.mean()))
        summary.setdefault("beta_shift_mean", float(beta_shift.mean()))
    else:
        summary.setdefault("beta_mean", float("nan"))
        summary.setdefault("beta_shift_mean", float("nan"))
    payload["summary"] = summary

    return payload


def export_adaptive_report(results: dict[str, Any]) -> Path:
    """Create a Markdown report with the adaptive simulation outcome."""

    if not isinstance(results, dict):
        raise ValueError("Se requieren resultados en formato dict para exportar el reporte adaptativo")

    summary = results.get("summary") if isinstance(results.get("summary"), dict) else {}
    steps = results.get("steps")
    steps_df = steps.copy() if isinstance(steps, pd.DataFrame) else pd.DataFrame(steps or [])
    if not steps_df.empty:
        steps_df = steps_df.copy()
        steps_df["timestamp"] = pd.to_datetime(steps_df.get("timestamp"), errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        steps_df = steps_df.sort_values("timestamp").tail(30)

    reports_dir = Path("docs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.utcnow()
    filename = f"adaptive_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
    report_path = reports_dir / filename

    def _fmt_percent(value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "-"
        if not np.isfinite(numeric):
            return "-"
        return f"{numeric:.2f}%"

    def _fmt_float(value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "-"
        if not np.isfinite(numeric):
            return "-"
        return f"{numeric:.2f}"

    beta_avg = _fmt_float(summary.get("beta_shift_avg")) if summary else "-"
    sigma_sector = _fmt_percent(summary.get("sector_dispersion")) if summary else "-"

    metrics_section = "\n".join(
        [
            "## Resumen global de métricas",
            f"- MAE adaptativo: {_fmt_percent(summary.get('mae')) if summary else '-'}",
            f"- RMSE adaptativo: {_fmt_percent(summary.get('rmse')) if summary else '-'}",
            f"- Bias adaptativo: {_fmt_percent(summary.get('bias')) if summary else '-'}",
            f"- β-shift promedio: {beta_avg}",
            f"- σ sectorial: {sigma_sector}",
        ]
    )

    if steps_df.empty:
        timeline_section = "## Tabla temporal de evolución\n_Sin registros disponibles._"
    else:
        try:
            table_text = steps_df.to_markdown(index=False)
        except (ImportError, ValueError):
            table_text = steps_df.to_csv(index=False, sep="|")
        timeline_section = "\n".join(
            [
                "## Tabla temporal de evolución",
                table_text,
            ]
        )

    interpretation_lines = [
        "## Interpretación del β-shift y dispersión sectorial",
        "El β-shift promedio refleja la magnitud media de los ajustes aplicados en cada iteración del modelo.",
        "Una mayor σ sectorial indica mayor heterogeneidad en los retornos proyectados entre sectores, guiando la diversificación.",
    ]
    if summary and isinstance(summary, dict):
        interpretation_lines.append(f"Resumen generado: {summary.get('text', '')}")

    content = "\n\n".join(
        [
            f"# Reporte adaptativo ({timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC)",
            metrics_section,
            timeline_section,
            "\n".join(interpretation_lines),
        ]
    )

    report_path.write_text(content, encoding="utf-8")
    LOGGER.debug("Reporte adaptativo exportado en %s", report_path)
    return report_path


__all__ = [
    "generate_synthetic_history",
    "prepare_adaptive_history",
    "simulate_adaptive_forecast",
    "export_adaptive_report",
    "update_model",
]
