"""Engine service endpoints for predictive infrastructure."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import logging
from typing import Any, Mapping

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field

from application.backtesting_service import BacktestingService
from application.predictive_core import (
    average_correlation,
    compute_ema_prediction,
    extract_backtest_series,
    run_backtest,
)
from predictive_engine.adapters import run_adaptive_forecast
from predictive_engine.base import compute_sector_predictions
from predictive_engine.models import AdaptiveForecastResult, AdaptiveUpdateResult, SectorPredictionSet
from predictive_engine.storage import load_forecast_history
from predictive_engine.utils import series_to_dict, to_native, to_records
from services.auth import get_current_user
from services.performance_metrics import measure_execution
from shared.version import __build_signature__, __version__

from api.routers.base_models import _BaseModel
from api.schemas.adaptive_utils import validate_adaptive_limits
from api.schemas.predictive import (
    AdaptiveForecastRequest as BaseAdaptiveForecastRequest,
    AdaptiveHistoryEntry,
    OpportunityPayload,
    PredictRequest,
    PredictResponse as BasePredictResponse,
    SectorPrediction,
)

logger = logging.getLogger(__name__)
logger.info("Initialising engine router")

def _model_dump(model: _BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _coerce_timestamp(value: datetime | str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        ts = pd.Timestamp(value)
    else:
        ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    tz_info = getattr(ts, "tzinfo", None)
    if tz_info is not None:
        try:
            return ts.tz_localize(None)
        except (TypeError, AttributeError):
            try:
                return ts.tz_convert(None)
            except (TypeError, AttributeError):
                try:
                    return pd.Timestamp(ts.to_pydatetime().replace(tzinfo=None))
                except Exception:  # pragma: no cover - defensive fallback
                    return ts
    return ts


def _build_frame(entries: Sequence[_BaseModel], columns: Sequence[str]) -> pd.DataFrame | None:
    if not entries:
        return None
    data = [_model_dump(entry) for entry in entries]
    frame = pd.DataFrame(data)
    if frame.empty:
        return None
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    frame = frame.loc[:, list(columns)].copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = frame["timestamp"].apply(_coerce_timestamp)
    return frame


def _frame_or_none(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    return frame


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


router = APIRouter(prefix="/engine", tags=["engine"])


class PredictResponse(BasePredictResponse):
    """Engine predictive response wrapper."""


class AdaptivePredictionEntry(_BaseModel):
    timestamp: datetime | str | None = Field(default=None, description="Timestamp for the prediction.")
    sector: str
    predicted_return: float


class AdaptiveActualEntry(_BaseModel):
    timestamp: datetime | str | None = Field(default=None, description="Timestamp for the actual return.")
    sector: str
    actual_return: float


class AdaptiveForecastRequest(BaseAdaptiveForecastRequest):
    predictions: Sequence[AdaptivePredictionEntry] = Field(default_factory=list)
    actuals: Sequence[AdaptiveActualEntry] = Field(default_factory=list)
    max_history_rows: int = Field(720, ge=1, description="Maximum history rows to persist in memory.")
    persist: bool = Field(True, description="Whether to persist adaptive state between requests.")
    persist_history: bool = Field(False, description="Whether to persist history snapshots to disk.")
    history_path: str | None = Field(
        "./data/forecast_history.parquet",
        description="Filesystem path used when persisting adaptive history.",
    )
    warm_start: bool = Field(True, description="Warm start using persisted history when available.")
    timestamp: datetime | str | None = Field(
        default=None,
        description="Optional timestamp override for incremental updates.",
    )



class AdaptiveUpdateResponse(_BaseModel):
    timestamp: datetime | str | None = None
    normalized: Sequence[dict[str, Any]] = Field(default_factory=list)
    history: Sequence[dict[str, Any]] = Field(default_factory=list)
    correlation_matrix: Sequence[dict[str, Any]] = Field(default_factory=list)
    beta_shift: dict[str, Any] = Field(default_factory=dict)


class AdaptiveForecastPayload(_BaseModel):
    mae: float = 0.0
    rmse: float = 0.0
    bias: float = 0.0
    raw_mae: float = 0.0
    raw_rmse: float = 0.0
    raw_bias: float = 0.0
    beta_shift_avg: float = 0.0
    sector_dispersion: float = 0.0
    correlation_mean: float | None = None
    beta_shift: dict[str, Any] = Field(default_factory=dict)
    correlation_matrix: Sequence[dict[str, Any]] = Field(default_factory=list)
    historical_correlation: Sequence[dict[str, Any]] = Field(default_factory=list)
    rolling_correlation: Sequence[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    steps: Sequence[dict[str, Any]] = Field(default_factory=list)


class AdaptiveForecastResponse(_BaseModel):
    cache_hit: bool = False
    cache_metadata: dict[str, Any] = Field(default_factory=dict)
    forecast: AdaptiveForecastPayload | None = None
    update: AdaptiveUpdateResponse | None = None


class HistoryResponse(_BaseModel):
    history: Sequence[dict[str, Any]] = Field(default_factory=list)


@router.get("/info", summary="Engine service metadata")
async def engine_info() -> dict[str, str]:
    """Return static metadata describing the predictive engine."""

    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "status": "ok",
        "engine_version": f"v{__version__}",
        "build_signature": __build_signature__,
        "timestamp": timestamp,
    }


@router.post("/predict", response_model=PredictResponse, summary="Compute sector predictions")
async def engine_predict(
    payload: PredictRequest,
    _claims: dict = Depends(get_current_user),
) -> PredictResponse:
    """Execute the sector prediction engine using EMA-smoothed backtests."""

    logger.info(
        "Running sector prediction request (items=%d, span=%d)",
        len(payload.opportunities),
        payload.span,
    )
    opportunity_rows = [_model_dump(entry) for entry in payload.opportunities]
    frame = pd.DataFrame(opportunity_rows) if opportunity_rows else None
    if frame is None or frame.empty or not {"symbol", "sector"}.issubset(frame.columns):
        logger.info("No valid opportunities received; returning empty prediction set")
        return PredictResponse()

    frame = frame.loc[:, ["symbol", "sector"]].copy()

    service = BacktestingService()
    with measure_execution("engine.predict"):
        result = compute_sector_predictions(
            frame,
            backtesting_service=service,
            run_backtest=lambda svc, symbol: run_backtest(svc, symbol, logger=logger),
            extract_series=extract_backtest_series,
            ema_predictor=lambda series, ema_span: compute_ema_prediction(series, span=ema_span),
            average_correlation=average_correlation,
            span=payload.span,
            logger=logger,
        )

    if isinstance(result, SectorPredictionSet):
        predictions_frame = result.to_dataframe()
    elif isinstance(result, pd.DataFrame):
        predictions_frame = result.copy()
    else:
        predictions_frame = pd.DataFrame(result or [])

    records = to_records(predictions_frame)
    parsed = [SectorPrediction(**record) for record in records]
    return PredictResponse(predictions=parsed)


@router.post(
    "/forecast/adaptive",
    response_model=AdaptiveForecastResponse,
    summary="Run adaptive forecast workflow",
)
async def engine_forecast_adaptive(
    payload: AdaptiveForecastRequest,
    _claims: dict = Depends(get_current_user),
) -> AdaptiveForecastResponse:
    """Execute the adaptive forecasting routine and expose serialised outputs."""

    logger.info(
        "Running adaptive forecast (history=%d, predictions=%d, actuals=%d)",
        len(payload.history),
        len(payload.predictions),
        len(payload.actuals),
    )

    validate_adaptive_limits(payload.history, max_size=payload.max_history_rows)

    history_frame = _frame_or_none(
        _build_frame(payload.history, ["timestamp", "sector", "predicted_return", "actual_return"])
    )
    predictions_frame = _frame_or_none(
        _build_frame(payload.predictions, ["timestamp", "sector", "predicted_return"])
    )
    actuals_frame = _frame_or_none(
        _build_frame(payload.actuals, ["timestamp", "sector", "actual_return"])
    )

    if history_frame is not None and (predictions_frame is not None or actuals_frame is not None):
        raise HTTPException(
            status_code=400,
            detail="history and (predictions, actuals) are mutually exclusive inputs.",
        )

    timestamp_override = _coerce_timestamp(payload.timestamp)

    with measure_execution("engine.forecast_adaptive"):
        result = run_adaptive_forecast(
            history=history_frame,
            predictions=predictions_frame,
            actuals=actuals_frame,
            ema_span=payload.ema_span,
            rolling_window=payload.rolling_window,
            ttl_hours=payload.ttl_hours,
            max_history_rows=payload.max_history_rows,
            persist_state=payload.persist,
            persist_history=payload.persist_history,
            history_path=payload.history_path,
            warm_start=payload.warm_start,
            timestamp=timestamp_override,
        )

    forecast_payload: AdaptiveForecastPayload | None = None
    forecast_result = result.get("forecast")
    if isinstance(forecast_result, AdaptiveForecastResult):
        forecast_data = forecast_result.as_dict()
    elif isinstance(forecast_result, dict):
        forecast_data = forecast_result
    else:
        forecast_data = None

    if forecast_data:
        summary_native = to_native(forecast_data.get("summary", {}))
        if not isinstance(summary_native, dict):
            summary_native = {}
        forecast_payload = AdaptiveForecastPayload(
            mae=_coerce_float(forecast_data.get("mae")),
            rmse=_coerce_float(forecast_data.get("rmse")),
            bias=_coerce_float(forecast_data.get("bias")),
            raw_mae=_coerce_float(forecast_data.get("raw_mae")),
            raw_rmse=_coerce_float(forecast_data.get("raw_rmse")),
            raw_bias=_coerce_float(forecast_data.get("raw_bias")),
            beta_shift_avg=_coerce_float(forecast_data.get("beta_shift_avg")),
            sector_dispersion=_coerce_float(forecast_data.get("sector_dispersion")),
            correlation_mean=forecast_data.get("correlation_mean"),
            beta_shift=series_to_dict(forecast_data.get("beta_shift")),
            correlation_matrix=to_records(forecast_data.get("correlation_matrix")),
            historical_correlation=to_records(forecast_data.get("historical_correlation")),
            rolling_correlation=to_records(forecast_data.get("rolling_correlation")),
            summary=summary_native,
            steps=to_records(forecast_data.get("steps")),
        )

    update_payload: AdaptiveUpdateResponse | None = None
    update_result = result.get("update")
    if isinstance(update_result, AdaptiveUpdateResult):
        update_data = update_result.to_dict()
    elif isinstance(update_result, dict):
        update_data = update_result
    else:
        update_data = None

    if update_data:
        update_payload = AdaptiveUpdateResponse(
            timestamp=to_native(update_data.get("timestamp")),
            normalized=to_records(update_data.get("normalized")),
            history=to_records(update_data.get("history")),
            correlation_matrix=to_records(update_data.get("correlation_matrix")),
            beta_shift=series_to_dict(update_data.get("beta_shift")),
        )

    cache_metadata = to_native(result.get("cache_metadata", {})) or {}

    return AdaptiveForecastResponse(
        cache_hit=bool(result.get("cache_hit", False)),
        cache_metadata=cache_metadata if isinstance(cache_metadata, dict) else {},
        forecast=forecast_payload,
        update=update_payload,
    )


@router.get("/history", response_model=HistoryResponse, summary="Retrieve adaptive forecast history")
async def engine_history(
    _claims: dict = Depends(get_current_user),
) -> HistoryResponse:
    """Return the persisted adaptive forecast history if available."""

    logger.info("Fetching adaptive forecast history")
    with measure_execution("engine.history"):
        history_frame = load_forecast_history()
    records = to_records(history_frame)
    return HistoryResponse(history=records)
