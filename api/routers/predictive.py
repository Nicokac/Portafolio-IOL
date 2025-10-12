"""Predictive analytics endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

try:  # pragma: no cover - compatibility shim for Pydantic v1/v2
    from pydantic import model_validator
except ImportError:  # pragma: no cover
    model_validator = None  # type: ignore[assignment]
    from pydantic import root_validator  # type: ignore[attr-defined]
else:  # pragma: no cover - ensure name exists for type checkers
    root_validator = None  # type: ignore[assignment]

from application.adaptive_predictive_service import simulate_adaptive_forecast
from application.predictive_service import predict_sector_performance

router = APIRouter(prefix="/predictive", tags=["predictive"])


try:  # pragma: no cover - compatibility shim for Pydantic v1/v2
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover
    ConfigDict = None  # type: ignore[assignment]


class _BaseModel(BaseModel):
    """Base model that tolerates extra fields across Pydantic versions."""

    if ConfigDict is not None:  # pragma: no branch
        model_config = ConfigDict(extra="ignore")  # type: ignore[assignment]
    else:  # pragma: no cover - fallback for Pydantic v1
        class Config:  # type: ignore[override]
            extra = "ignore"


class OpportunityPayload(_BaseModel):
    """Minimal data required to evaluate sector opportunities."""

    symbol: str = Field(..., description="Ticker or instrument symbol")
    sector: str = Field(..., description="Associated sector for the instrument")


class PredictRequest(_BaseModel):
    """Request payload for sector performance predictions."""

    opportunities: Sequence[OpportunityPayload] = Field(
        default_factory=list,
        description="Universe of instruments grouped by sector.",
    )
    span: int = Field(10, ge=1, description="EMA span applied during smoothing.")
    ttl_hours: float | None = Field(
        default=None,
        ge=0.0,
        description="Override TTL for cached predictions in hours.",
    )


class SectorPrediction(_BaseModel):
    """Normalised prediction data for a sector."""

    sector: str
    predicted_return: float
    sample_size: int | None = None
    avg_correlation: float | None = None
    confidence: float | None = None


class PredictResponse(_BaseModel):
    """Response returned by the predictive endpoint."""

    predictions: Sequence[SectorPrediction] = Field(default_factory=list)


class AdaptiveHistoryEntry(_BaseModel):
    """Historical observation for adaptive forecasting."""

    timestamp: datetime | str | None = Field(
        default=None,
        description="Timestamp of the observation.",
    )
    sector: str
    predicted_return: float
    actual_return: float
    symbol: str | None = Field(
        default=None,
        description="Optional symbol associated with the observation.",
    )


class AdaptiveForecastRequest(_BaseModel):
    """Request payload for adaptive forecast simulations."""

    history: Sequence[AdaptiveHistoryEntry] = Field(
        default_factory=list,
        description="Historical sector observations.",
    )
    ema_span: int = Field(4, ge=1, description="EMA span for adaptive updates.")
    persist: bool = Field(
        True,
        description="Whether to persist adaptive state across runs.",
    )
    rolling_window: int = Field(
        20,
        ge=2,
        description="Window size for rolling correlation measurements.",
    )
    ttl_hours: float | None = Field(
        default=None,
        ge=0.0,
        description="Override TTL for adaptive cache metadata in hours.",
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

    else:  # pragma: no cover - fallback for Pydantic v1

        @root_validator(skip_on_failure=True)  # type: ignore[misc]
        def _check_limits_v1(
            cls, values: dict[str, Any]
        ) -> dict[str, Any]:
            cls._validate_limits(values)
            return values


class AdaptiveForecastResponse(_BaseModel):
    """Normalised response for adaptive forecast simulations."""

    mae: float = 0.0
    rmse: float = 0.0
    bias: float = 0.0
    raw_mae: float = 0.0
    raw_rmse: float = 0.0
    raw_bias: float = 0.0
    beta_shift: Mapping[str, float] = Field(default_factory=dict)
    correlation_matrix: Sequence[Mapping[str, Any]] = Field(default_factory=list)
    historical_correlation: Sequence[Mapping[str, Any]] = Field(default_factory=list)
    rolling_correlation: Sequence[Mapping[str, Any]] = Field(default_factory=list)
    summary: Mapping[str, Any] = Field(default_factory=dict)
    steps: Sequence[Mapping[str, Any]] = Field(default_factory=list)
    cache_metadata: Mapping[str, Any] = Field(default_factory=dict)


def _to_primitive(value: Any) -> Any:
    """Convert pandas/numpy objects into JSON-serialisable primitives."""

    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (np.ndarray,)):
        return [_to_primitive(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _to_primitive(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_primitive(item) for item in value]
    return value


def _frame_to_records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    records = frame.reset_index(drop=False).to_dict(orient="records")
    return [
        {str(key): _to_primitive(value) for key, value in record.items()}
        for record in records
    ]


def _series_to_mapping(series: pd.Series | None) -> dict[str, Any]:
    if not isinstance(series, pd.Series) or series.empty:
        return {}
    return {
        str(index): _to_primitive(value)
        for index, value in series.items()
        if pd.notna(value)
    }


@router.get("/", summary="Predictive service placeholder")
async def predictive_root() -> dict[str, str]:
    """Placeholder endpoint for predictive services."""
    return {"detail": "Predictive endpoints coming soon."}


@router.post("/predict", response_model=PredictResponse, summary="Predict sector performance")
async def predict_sector(payload: PredictRequest) -> PredictResponse:
    """Run the predictive engine and expose sector-level expectations."""

    data = [entry.dict() for entry in payload.opportunities]
    frame = pd.DataFrame(data) if data else None
    predictions = predict_sector_performance(
        frame,
        span=payload.span,
        ttl_hours=payload.ttl_hours,
    )
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions or [])
    records = predictions.to_dict(orient="records")
    parsed = [SectorPrediction(**record) for record in records]
    return PredictResponse(predictions=parsed)


@router.post(
    "/forecast/adaptive",
    response_model=AdaptiveForecastResponse,
    summary="Simulate adaptive forecast",
)
async def forecast_adaptive(payload: AdaptiveForecastRequest) -> AdaptiveForecastResponse:
    """Execute the adaptive forecasting workflow and normalise the output."""

    history = [entry.dict() for entry in payload.history]
    frame = pd.DataFrame(history) if history else None
    result = simulate_adaptive_forecast(
        frame,
        ema_span=payload.ema_span,
        persist=payload.persist,
        rolling_window=payload.rolling_window,
        ttl_hours=payload.ttl_hours,
    )
    if not isinstance(result, Mapping):
        result = {}
    response = AdaptiveForecastResponse(
        mae=float(_to_primitive(result.get("mae", 0.0)) or 0.0),
        rmse=float(_to_primitive(result.get("rmse", 0.0)) or 0.0),
        bias=float(_to_primitive(result.get("bias", 0.0)) or 0.0),
        raw_mae=float(_to_primitive(result.get("raw_mae", 0.0)) or 0.0),
        raw_rmse=float(_to_primitive(result.get("raw_rmse", 0.0)) or 0.0),
        raw_bias=float(_to_primitive(result.get("raw_bias", 0.0)) or 0.0),
        beta_shift=_series_to_mapping(result.get("beta_shift")),
        correlation_matrix=_frame_to_records(result.get("correlation_matrix")),
        historical_correlation=_frame_to_records(result.get("historical_correlation")),
        rolling_correlation=_frame_to_records(result.get("rolling_correlation")),
        summary=_to_primitive(result.get("summary", {})),
        steps=_frame_to_records(result.get("steps")),
        cache_metadata=_to_primitive(result.get("cache_metadata", {})),
    )
    return response
