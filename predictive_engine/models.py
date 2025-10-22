"""Domain models employed by the predictive engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import pandas as pd


@dataclass(frozen=True)
class SectorPrediction:
    """Single sector level prediction row."""

    sector: str
    predicted_return: float
    sample_size: int
    avg_correlation: float
    confidence: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "sector": self.sector,
            "predicted_return": float(self.predicted_return),
            "sample_size": int(self.sample_size),
            "avg_correlation": float(self.avg_correlation),
            "confidence": float(self.confidence),
        }


@dataclass
class SectorPredictionSet:
    """Collection of :class:`SectorPrediction` rows."""

    rows: List[SectorPrediction] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(
                columns=[
                    "sector",
                    "predicted_return",
                    "sample_size",
                    "avg_correlation",
                    "confidence",
                ],
            )
        frame = pd.DataFrame([row.to_dict() for row in self.rows])
        return frame.sort_values("predicted_return", ascending=False).reset_index(
            drop=True
        )

    def __iter__(self) -> Iterable[SectorPrediction]:
        return iter(self.rows)

    @property
    def is_empty(self) -> bool:
        return len(self.rows) == 0


@dataclass(frozen=True)
class ModelMetrics:
    """Collection of error and dispersion metrics for the adaptive engine."""

    mae: float
    rmse: float
    bias: float
    raw_mae: float
    raw_rmse: float
    raw_bias: float
    beta_shift_avg: float
    correlation_mean: float
    sector_dispersion: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mae": float(self.mae),
            "rmse": float(self.rmse),
            "bias": float(self.bias),
            "raw_mae": float(self.raw_mae),
            "raw_rmse": float(self.raw_rmse),
            "raw_bias": float(self.raw_bias),
            "beta_shift_avg": float(self.beta_shift_avg),
            "correlation_mean": float(self.correlation_mean),
            "sector_dispersion": float(self.sector_dispersion),
        }


@dataclass
class CorrelationBundle:
    """Group of correlation matrices captured by the adaptive engine."""

    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    historical_correlation: pd.DataFrame = field(default_factory=pd.DataFrame)
    rolling_correlation: pd.DataFrame = field(default_factory=pd.DataFrame)

    def copy(self) -> "CorrelationBundle":
        return CorrelationBundle(
            correlation_matrix=self.correlation_matrix.copy(),
            historical_correlation=self.historical_correlation.copy(),
            rolling_correlation=self.rolling_correlation.copy(),
        )


@dataclass
class AdaptiveState:
    """State persisted between adaptive updates."""

    history: pd.DataFrame
    last_updated: pd.Timestamp | None = None

    def copy(self) -> "AdaptiveState":
        return AdaptiveState(
            history=self.history.copy(),
            last_updated=self.last_updated,
        )


@dataclass
class AdaptiveUpdateResult:
    """Outcome of an adaptive state update iteration."""

    state: AdaptiveState
    normalized: pd.DataFrame
    correlation_matrix: pd.DataFrame
    beta_shift: pd.Series
    timestamp: pd.Timestamp | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "normalized": self.normalized.copy(),
            "history": self.state.history.copy(),
            "correlation_matrix": self.correlation_matrix.copy(),
            "beta_shift": self.beta_shift.copy(),
        }


@dataclass
class AdaptiveForecastResult:
    """Aggregate payload returned by :func:`calculate_adaptive_forecast`."""

    metrics: ModelMetrics
    beta_shift: pd.Series
    correlations: CorrelationBundle
    steps: pd.DataFrame
    cache_metadata: Dict[str, Any]
    summary: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "mae": self.metrics.mae,
            "rmse": self.metrics.rmse,
            "bias": self.metrics.bias,
            "raw_mae": self.metrics.raw_mae,
            "raw_rmse": self.metrics.raw_rmse,
            "raw_bias": self.metrics.raw_bias,
            "beta_shift": self.beta_shift.copy(),
            "correlation_matrix": self.correlations.correlation_matrix.copy(),
            "historical_correlation": self.correlations.historical_correlation.copy(),
            "rolling_correlation": self.correlations.rolling_correlation.copy(),
            "summary": dict(self.summary),
            "steps": self.steps.copy(),
            "cache_metadata": dict(self.cache_metadata),
        }
        payload.update(self.metrics.to_dict())
        return payload


def empty_history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["timestamp", "sector", "predicted_return", "actual_return"]
    )
