from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from application.backtesting_service import BacktestingService
from application.predictive_core import PredictiveCacheState
from application.predictive_service import build_adaptive_history
from services.cache import CacheService
from services.performance_metrics import track_function
from services.performance_timer import ProfileBlockResult, profile_block
from shared.settings import ADAPTIVE_TTL_HOURS

from predictive_engine import __version__ as ENGINE_VERSION
from predictive_engine.adapters import run_adaptive_forecast
from predictive_engine.models import AdaptiveUpdateResult, AdaptiveState, empty_history_frame
from predictive_engine import utils as engine_utils
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
_DEFAULT_BATCH_SIZE = 10
_MAX_BATCH_WORKERS = 4


def _coerce_adaptive_state(value: Any) -> AdaptiveState:
    if isinstance(value, AdaptiveState):
        return value.copy()
    history = empty_history_frame()
    last_updated = getattr(value, "last_updated", None)
    maybe_history = getattr(value, "history", None)
    if isinstance(value, dict):
        maybe_history = value.get("history", maybe_history)
        last_updated = value.get("last_updated", last_updated)
    if isinstance(maybe_history, pd.DataFrame):
        history = maybe_history.copy()
    return AdaptiveState(history=history, last_updated=last_updated)


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


def _resolve_identifier_column(
    predictions: pd.DataFrame | None, actuals: pd.DataFrame | None
) -> str | None:
    candidates = ("ticker", "symbol", "isin", "asset", "sector")
    for column in candidates:
        for frame in (predictions, actuals):
            if isinstance(frame, pd.DataFrame) and column in frame.columns:
                return column
    return None


def _build_prediction_batches(
    predictions: pd.DataFrame | None,
    actuals: pd.DataFrame | None,
    *,
    batch_size: int,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    batch_size = max(int(batch_size), 1)
    identifier = _resolve_identifier_column(predictions, actuals)
    pred_frame = predictions.copy() if isinstance(predictions, pd.DataFrame) else pd.DataFrame()
    act_frame = actuals.copy() if isinstance(actuals, pd.DataFrame) else pd.DataFrame()

    if identifier is None:
        return [(0, pred_frame, act_frame)]

    seen: set[Any] = set()
    identifiers: list[Any] = []
    for frame in (pred_frame, act_frame):
        if identifier not in frame.columns:
            continue
        for value in frame[identifier].tolist():
            if pd.isna(value) or value in seen:
                continue
            seen.add(value)
            identifiers.append(value)

    if not identifiers:
        return [(0, pred_frame, act_frame)]

    batches: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    for index, start in enumerate(range(0, len(identifiers), batch_size)):
        batch_ids = identifiers[start : start + batch_size]
        preds_batch = (
            pred_frame[pred_frame[identifier].isin(batch_ids)].copy()
            if not pred_frame.empty
            else pd.DataFrame(columns=pred_frame.columns)
        )
        acts_batch = (
            act_frame[act_frame[identifier].isin(batch_ids)].copy()
            if not act_frame.empty
            else pd.DataFrame(columns=act_frame.columns)
        )
        batches.append((index, preds_batch, acts_batch))
    return batches


def _normalize_batch(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    timestamp: pd.Timestamp | None,
) -> pd.DataFrame:
    normalized_predictions = engine_utils.normalise_predictions(predictions)
    normalized_actuals = engine_utils.normalise_actuals(actuals)
    merged = engine_utils.merge_inputs(
        normalized_predictions,
        normalized_actuals,
        timestamp=timestamp,
    )
    return engine_utils.prepare_normalized_frame(merged)


def _process_prediction_batches(
    batches: Sequence[tuple[int, pd.DataFrame, pd.DataFrame]],
    *,
    timestamp: pd.Timestamp | None,
) -> tuple[pd.DataFrame, int]:
    if not batches:
        empty = pd.DataFrame(columns=["timestamp", "sector", "normalized_error"])
        return empty, 0

    workers = max(1, min(_MAX_BATCH_WORKERS, len(batches)))
    results: list[tuple[int, pd.DataFrame]] = []
    successes = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_normalize_batch, preds, acts, timestamp=timestamp): index
            for index, preds, acts in batches
        }
        for future in as_completed(futures):
            batch_index = futures[future]
            normalized = future.result()
            results.append((batch_index, normalized))
            successes += 1

    results.sort(key=lambda item: item[0])
    frames = [frame for _, frame in results if isinstance(frame, pd.DataFrame) and not frame.empty]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=["timestamp", "sector", "normalized_error"])
    return combined, successes


@contextmanager
def _adaptive_lock_scope(
    operation: str,
    *,
    timeout_s: float,
    extra: dict[str, str] | None = None,
) -> Iterator[None]:
    timeout_s = max(float(timeout_s), 0.0)
    while True:
        acquired = adaptive_cache_lock.acquire_with_timeout(timeout_s)
        if not acquired:
            LOGGER.warning(
                "Timeout al adquirir lock adaptativo para %s tras %.1fs; reintentando",
                operation,
                timeout_s,
            )
            time.sleep(min(timeout_s / 4.0 if timeout_s else 0.1, 1.0))
            continue
        lock_profile: ProfileBlockResult | None = None
        try:
            with profile_block(
                f"adaptive_predictive.lock_scope.{operation}",
                extra=extra or {"operation": operation},
                module=__name__,
                threshold_s=_LOCK_PROLONGED_THRESHOLD_S,
            ) as lock_profile_ctx:
                lock_profile = lock_profile_ctx
                yield
            break
        finally:
            adaptive_cache_lock.release()
            _warn_prolonged_lock(lock_profile, operation=operation)


def _write_performance_metrics(runtime_s: float, success_pct: float) -> None:
    metrics_path = Path("performance_metrics_8.csv")
    lines = [
        "metric,value,notes",
        (
            "recommendations.predictive_runtime_s,"
            f"{max(runtime_s, 0.0):.4f},\"Duración total del procesamiento adaptativo\""
        ),
        (
            "recommendations.batch_success_rate_pct,"
            f"{max(success_pct, 0.0):.2f},\"Porcentaje de sub-batches procesados con éxito\""
        ),
    ]
    metrics_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_model(
    predictions: pd.DataFrame | None,
    actuals: pd.DataFrame | None,
    *,
    cache: CacheService | None = None,
    ema_span: int = _DEFAULT_EMA_SPAN,
    timestamp: pd.Timestamp | None = None,
    persist: bool = True,
    ttl_hours: float | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    lock_timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Update the adaptive state using normalized prediction errors."""

    LOGGER.debug("Ejecutando predictive engine %s", ENGINE_VERSION)

    effective_ttl_hours = (
        float(ttl_hours)
        if ttl_hours is not None
        else float(ADAPTIVE_TTL_HOURS)
    )

    if persist:
        active_cache = cache or _CACHE
    else:
        active_cache = cache or CacheService(namespace=f"{_CACHE_NAMESPACE}_ephemeral")
    ttl_seconds = max(effective_ttl_hours, 0.0) * 3600.0
    if isinstance(active_cache, CacheService):
        active_cache.set_ttl_override(ttl_seconds)

    overall_start = time.perf_counter()

    batches = _build_prediction_batches(
        predictions,
        actuals,
        batch_size=batch_size,
    )
    total_batches = len(batches) or 1

    cache_hit = False
    base_state = AdaptiveState(history=empty_history_frame(), last_updated=None)

    LOGGER.debug(
        "Intentando adquirir lock para fetch adaptativo (ema_span=%s, persist=%s)",
        ema_span,
        persist,
    )
    with _adaptive_lock_scope(
        "update.fetch",
        timeout_s=lock_timeout_s,
        extra={
            "operation": "update_model",
            "stage": "fetch",
            "ema_span": str(ema_span),
        },
    ):
        cached_state = active_cache.get(_STATE_KEY)
        cache_hit = cached_state is not None
        if cached_state is not None:
            try:
                base_state = _coerce_adaptive_state(cached_state)
            except Exception:  # pragma: no cover - defensive
                base_state = AdaptiveState(history=empty_history_frame(), last_updated=None)

    normalized_frame, successful_batches = _process_prediction_batches(
        batches,
        timestamp=timestamp,
    )

    history, last_timestamp = engine_utils.append_history(
        base_state.history,
        normalized_frame,
        max_rows=_MAX_HISTORY_ROWS,
    )
    pivot = engine_utils.pivot_history(history)
    correlation_matrix, beta_shift = engine_utils.compute_beta_shift(
        pivot,
        ema_span=ema_span,
    )

    updated_state = AdaptiveState(
        history=history,
        last_updated=last_timestamp or base_state.last_updated,
    )
    result_timestamp = timestamp or last_timestamp or base_state.last_updated
    update_result = AdaptiveUpdateResult(
        state=updated_state,
        normalized=normalized_frame,
        correlation_matrix=correlation_matrix.copy(),
        beta_shift=beta_shift.copy(),
        timestamp=result_timestamp,
    )

    LOGGER.debug(
        "Procesamiento adaptativo finalizado para %s batches (éxitos=%s)",
        total_batches,
        successful_batches,
    )

    cache_last_updated = _cache_last_updated(active_cache)

    if persist:
        LOGGER.debug("Persistiendo estado adaptativo (ttl=%.2fs)", ttl_seconds)
        with _adaptive_lock_scope(
            "update.persist",
            timeout_s=lock_timeout_s,
            extra={
                "operation": "update_model",
                "stage": "persist",
                "ema_span": str(ema_span),
            },
        ):
            try:
                active_cache.set(_STATE_KEY, update_result.state, ttl=ttl_seconds)
                active_cache.set(
                    _CORR_KEY,
                    update_result.correlation_matrix.copy(),
                    ttl=ttl_seconds,
                )
            except Exception:  # pragma: no cover - cache persistence best-effort
                LOGGER.exception("No se pudo persistir el estado adaptativo en cache")
            cache_last_updated = _cache_last_updated(active_cache)

    hit_ratio = 0.0
    if hasattr(active_cache, "hit_ratio"):
        try:
            hit_ratio = float(active_cache.hit_ratio())
        except Exception:  # pragma: no cover - telemetry guard
            hit_ratio = 0.0
    cache_metadata = {
        "hit_ratio": hit_ratio,
        "last_updated": cache_last_updated or "-",
    }

    cache_timestamp = cache_metadata.get("last_updated")
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

    runtime_s = time.perf_counter() - overall_start
    success_pct = (
        float(successful_batches) / float(total_batches) * 100.0
        if total_batches
        else 0.0
    )
    try:
        _write_performance_metrics(runtime_s, success_pct)
    except Exception:  # pragma: no cover - metrics should not break flow
        LOGGER.debug("No se pudo actualizar performance_metrics_8.csv", exc_info=True)

    payload = update_result.to_dict()
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
    lock_timeout_s: float = 60.0,
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
        "Intentando adquirir lock para simulate_adaptive_forecast (ema_span=%s, persist=%s)",
        ema_span,
        persist,
    )
    with _adaptive_lock_scope(
        "forecast",
        timeout_s=lock_timeout_s,
        extra={
            "operation": "simulate_adaptive_forecast",
            "ema_span": str(ema_span),
            "persist": str(bool(persist)),
        },
    ):
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
