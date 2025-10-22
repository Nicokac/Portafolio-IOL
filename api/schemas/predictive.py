"""Shared schema definitions for predictive workflows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from pydantic import Field

try:  # pragma: no cover - compatibility shim for Pydantic v1/v2
    from pydantic import model_validator
except ImportError:  # pragma: no cover
    model_validator = None  # type: ignore[assignment]
    from pydantic import root_validator  # type: ignore[attr-defined]
else:  # pragma: no cover - ensure name exists for type checkers
    try:
        from pydantic import root_validator  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover
        root_validator = None  # type: ignore[assignment]

from api.routers.base_models import _BaseModel


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

    elif root_validator is not None:  # pragma: no cover - fallback for Pydantic v1

        @root_validator(skip_on_failure=True)  # type: ignore[misc]
        def _check_limits_v1(cls, values: dict[str, Any]) -> dict[str, Any]:
            cls._validate_limits(values)
            return values


__all__ = [
    "AdaptiveForecastRequest",
    "AdaptiveHistoryEntry",
    "OpportunityPayload",
    "PredictRequest",
    "PredictResponse",
    "SectorPrediction",
]
