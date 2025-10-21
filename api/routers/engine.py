"""Engine service endpoints for predictive infrastructure."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import logging
from typing import Any, Mapping

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

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

try:  # pragma: no cover - compatibility shim for Pydantic v1/v2
    from pydantic import Field, model_validator
except ImportError:  # pragma: no cover
    from pydantic import Field

    try:  # pragma: no cover - fallback for Pydantic v1
        from pydantic import root_validator  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover
        root_validator = None  # type: ignore[assignment]
    else:
        model_validator = None  # type: ignore[assignment]
else:  # pragma: no cover - ensure root_validator name exists for type checkers
    try:
        from pydantic import root_validator  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover
        root_validator = None  # type: ignore[assignment]

from api.routers.base_models import _BaseModel

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


class OpportunityPayload(_BaseModel):
    symbol: str
    sector: str


class PredictRequest(_BaseModel):
    opportunities: Sequence[OpportunityPayload] = Field(default_factory=list)
    span: int = Field(10, ge=1, description="EMA span applied during smoothing.")


class SectorPredictionRow(_BaseModel):
    sector: str
    predicted_return: float
    sample_size: int | None = None
    avg_correlation: float | None = None
    confidence: float | None = None


class PredictResponse(_BaseModel):
    predictions: Sequence[SectorPredictionRow] = Field(default_factory=list)


class AdaptiveHistoryEntry(_BaseModel):
    timestamp: datetime | str | None = Field(default=None, description="Timestamp for the observation.")
    sector: str
    predicted_return: float
    actual_return: float
    symbol: str | None = Field(
        default=None,
        description="Optional symbol associated with the observation.",
    )


class AdaptivePredictionEntry(_BaseModel):
    timestamp: datetime | str | None = Field(default=None, description="Timestamp for the prediction.")
    sector: str
    predicted_return: float


class AdaptiveActualEntry(_BaseModel):
    timestamp: datetime | str | None = Field(default=None, description="Timestamp for the actual return.")
    sector: str
    actual_return: float


class AdaptiveForecastRequest(_BaseModel):
    history: Sequence[AdaptiveHistoryEntry] = Field(default_factory=list)
    predictions: Sequence[AdaptivePredictionEntry] = Field(default_factory=list)
    actuals: Sequence[AdaptiveActualEntry] = Field(default_factory=list)
    ema_span: int = Field(5, ge=1, description="EMA span for adaptive updates.")
    rolling_window: int = Field(20, ge=2, description="Window size for rolling correlations.")
    ttl_hours: float | None = Field(default=None, ge=0.0, description="Override TTL for cached adaptive state.")
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

    @classmethod
    def _validate_limits(
        cls, data: "AdaptiveForecastRequest" | dict[str, Any]
    ) -> "AdaptiveForecastRequest" | dict[str, Any]:
        if isinstance(data, cls):
            history_items = list(data.history or [])
        else:
            history_items = list(data.get("history", []) or [])

        if len(history_items) > 10_000:
            raise ValueError("Se admiten hasta 10 000 observaciones en history")

        symbols: set[str] = set()
        for entry in history_items:
            symbol = getattr(entry, "symbol", None)
            if symbol is None and isinstance(entry, Mapping):
                symbol = entry.get("symbol")
            if symbol is not None:
                symbol_str = str(symbol).strip().upper()
                if symbol_str:
                    symbols.add(symbol_str)
                    continue
            sector = getattr(entry, "sector", None)
            if sector is None and isinstance(entry, Mapping):
                sector = entry.get("sector")
            sector_str = str(sector).strip().upper() if sector is not None else ""
            if sector_str:
                symbols.add(sector_str)

        if len(symbols) > 30:
            raise ValueError("Se admiten hasta 30 símbolos únicos en history")

        return data

    if model_validator is not None:  # pragma: no branch

        @model_validator(mode="after")
        def _check_limits(  # type: ignore[override]
            cls, model: "AdaptiveForecastRequest"
        ) -> "AdaptiveForecastRequest":
            cls._validate_limits(model)
            return model

    elif root_validator is not None:  # pragma: no cover - fallback for Pydantic v1

        @root_validator(skip_on_failure=True)  # type: ignore[misc]
        def _check_limits_v1(
            cls, values: dict[str, Any]
        ) -> dict[str, Any]:
            cls._validate_limits(values)
            return values


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
    parsed = [SectorPredictionRow(**record) for record in records]
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
