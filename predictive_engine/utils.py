"""Utility helpers for the predictive engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from data.normalizer import (
    ensure_timestamp,
    merge_inputs,
    normalise_actuals,
    normalise_predictions,
    prepare_normalized_frame,
)


def append_history(
    history: pd.DataFrame,
    normalized_rows: pd.DataFrame,
    *,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Append new normalised rows to history keeping a bounded size."""

    if normalized_rows.empty:
        return history.copy(), None
    frames = [frame for frame in (history, normalized_rows) if not frame.empty]
    if frames:
        frame = pd.concat(frames, ignore_index=True)
    else:
        frame = pd.DataFrame(columns=normalized_rows.columns)
    frame = frame.sort_values("timestamp")
    if len(frame) > max_rows:
        frame = frame.iloc[-max_rows:]
    frame = frame.reset_index(drop=True)
    last_timestamp = normalized_rows["timestamp"].iloc[-1]
    return frame, ensure_timestamp(last_timestamp)


def pivot_history(history: pd.DataFrame) -> pd.DataFrame:
    """Pivot the history into a timestamp-sector matrix."""

    if history.empty:
        return pd.DataFrame()
    pivot = history.pivot_table(
        index="timestamp",
        columns="sector",
        values="normalized_error",
        aggfunc="mean",
    )
    return pivot.sort_index()


def compute_beta_shift(
    pivot: pd.DataFrame,
    *,
    ema_span: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute the beta shift adjustments over the error pivot."""

    if pivot.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    span = max(int(ema_span), 1)
    smoothed = pivot.sort_index().ewm(span=span, adjust=False).mean()
    last_row = smoothed.iloc[-1]
    beta_shift = -last_row.fillna(0.0)
    corr = smoothed.corr().fillna(0.0)
    if not corr.empty:
        corr.values[np.diag_indices_from(corr.values)] = 1.0
    return corr, beta_shift


def mean_correlation(matrix: pd.DataFrame | None) -> float:
    """Average the upper triangular portion of a correlation matrix."""

    if matrix is None or matrix.empty:
        return float("nan")
    values = matrix.values
    if values.size == 0:
        return float("nan")
    upper = values[np.triu_indices_from(values, k=1)]
    upper = upper[np.isfinite(upper)]
    if upper.size == 0:
        diag = values[np.diag_indices_from(values)]
        diag = diag[np.isfinite(diag)]
        return float(diag.mean()) if diag.size else float("nan")
    return float(upper.mean())


def safe_mean(values: np.ndarray) -> float:
    """Return the arithmetic mean handling empty arrays gracefully."""

    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def safe_mae(values: np.ndarray) -> float:
    """Return the mean absolute error for ``values``."""

    if values.size == 0:
        return 0.0
    return float(np.mean(np.abs(values)))


def safe_rmse(values: np.ndarray) -> float:
    """Return the root mean squared error for ``values``."""

    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def compute_sector_dispersion(frame: pd.DataFrame) -> float:
    """Compute the dispersion of predicted returns across sectors."""

    if "predicted_return" not in frame.columns or frame.empty:
        return 0.0
    sector_means = (
        frame.groupby("sector")["predicted_return"]
        .mean()
        .replace([np.inf, -np.inf], np.nan)
    )
    if not isinstance(sector_means, pd.Series):
        return 0.0
    sector_means = sector_means.dropna()
    if sector_means.empty:
        return 0.0
    return float(sector_means.std(ddof=0))


def to_native(value: Any) -> Any:
    """Convert pandas/numpy objects into JSON-serialisable primitives."""

    if isinstance(value, pd.DataFrame):
        return to_records(value)
    if isinstance(value, pd.Series):
        return series_to_dict(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [to_native(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): to_native(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_native(item) for item in value]
    return value


def series_to_dict(series: pd.Series | None) -> dict[str, Any]:
    """Serialise a pandas Series into a mapping of native Python objects."""

    if not isinstance(series, pd.Series) or series.empty:
        return {}
    return {
        str(index): to_native(value)
        for index, value in series.items()
        if pd.notna(value)
    }


def to_records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Serialise a pandas DataFrame into a list of dictionaries."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    records = frame.reset_index(drop=False).to_dict(orient="records")
    return [
        {str(key): to_native(value) for key, value in record.items()}
        for record in records
    ]


__all__ = [
    "ensure_timestamp",
    "normalise_predictions",
    "normalise_actuals",
    "merge_inputs",
    "prepare_normalized_frame",
    "append_history",
    "pivot_history",
    "compute_beta_shift",
    "mean_correlation",
    "safe_mean",
    "safe_mae",
    "safe_rmse",
    "compute_sector_dispersion",
    "to_native",
    "series_to_dict",
    "to_records",
]
