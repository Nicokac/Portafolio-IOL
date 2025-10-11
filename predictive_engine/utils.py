"""Utility helpers for the predictive engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Tuple

import numpy as np
import pandas as pd


def ensure_timestamp(value: object) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value.tz_localize(None) if value.tzinfo else value
    if isinstance(value, str):
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return pd.Timestamp.utcnow().normalize()
        return parsed.tz_localize(None) if getattr(parsed, "tzinfo", None) else parsed
    if isinstance(value, (int, float)):
        parsed = pd.to_datetime(value, unit="s", errors="coerce")
        if pd.isna(parsed):
            return pd.Timestamp.utcnow().normalize()
        return parsed
    return pd.Timestamp.utcnow().normalize()


def normalise_predictions(frame: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=["sector", "predicted_return", "timestamp"])
    df = frame.copy()
    if "sector" not in df.columns:
        return pd.DataFrame(columns=["sector", "predicted_return", "timestamp"])
    df["sector"] = (
        df.get("sector", pd.Series(dtype=str))
        .astype("string")
        .str.strip()
        .replace({"": "Sin sector"})
    )
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(ensure_timestamp)
    else:
        df["timestamp"] = pd.Timestamp.utcnow().normalize()
    if "predicted_return" not in df.columns and "predicted_return_pct" in df.columns:
        df = df.rename(columns={"predicted_return_pct": "predicted_return"})
    if "predicted_return" not in df.columns:
        df["predicted_return"] = np.nan
    df["predicted_return"] = pd.to_numeric(df.get("predicted_return"), errors="coerce").astype(float)
    return df[["sector", "predicted_return", "timestamp"]]


def normalise_actuals(frame: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=["sector", "actual_return", "timestamp"])
    df = frame.copy()
    if "sector" not in df.columns:
        return pd.DataFrame(columns=["sector", "actual_return", "timestamp"])
    df["sector"] = (
        df.get("sector", pd.Series(dtype=str))
        .astype("string")
        .str.strip()
        .replace({"": "Sin sector"})
    )
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(ensure_timestamp)
    else:
        df["timestamp"] = pd.Timestamp.utcnow().normalize()
    if "actual_return" not in df.columns and "realized_return" in df.columns:
        df = df.rename(columns={"realized_return": "actual_return"})
    df["actual_return"] = pd.to_numeric(df.get("actual_return"), errors="coerce").astype(float)
    return df[["sector", "actual_return", "timestamp"]]


def merge_inputs(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    timestamp: pd.Timestamp | None,
) -> pd.DataFrame:
    if predictions.empty or actuals.empty:
        return pd.DataFrame(columns=["timestamp", "sector", "predicted_return", "actual_return"])
    merged = pd.merge(predictions, actuals, on=["sector", "timestamp"], how="outer")
    if timestamp is not None:
        merged["timestamp"] = ensure_timestamp(timestamp)
    else:
        merged["timestamp"] = merged["timestamp"].apply(ensure_timestamp)
    merged["predicted_return"] = pd.to_numeric(merged.get("predicted_return"), errors="coerce").astype(float)
    merged["actual_return"] = pd.to_numeric(merged.get("actual_return"), errors="coerce").astype(float)
    merged = merged.dropna(subset=["sector"])
    return merged


def prepare_normalized_frame(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame(columns=["timestamp", "sector", "normalized_error"])
    frame = merged.copy()
    frame = frame.dropna(subset=["sector", "predicted_return", "actual_return"])
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "sector", "normalized_error"])
    frame["timestamp"] = frame["timestamp"].apply(ensure_timestamp)
    frame = frame.sort_values("timestamp")
    frame["error"] = frame["predicted_return"] - frame["actual_return"]
    denominator = frame["actual_return"].abs().clip(lower=1e-4)
    frame["normalized_error"] = (frame["error"] / denominator).clip(-5.0, 5.0)
    return frame[["timestamp", "sector", "normalized_error"]]


def append_history(
    history: pd.DataFrame,
    normalized_rows: pd.DataFrame,
    *,
    max_rows: int,
) -> Tuple[pd.DataFrame, pd.Timestamp | None]:
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
    if history.empty:
        return pd.DataFrame()
    pivot = history.pivot_table(
        index="timestamp",
        columns="sector",
        values="normalized_error",
        aggfunc="mean",
    )
    return pivot.sort_index()


def compute_beta_shift(pivot: pd.DataFrame, *, ema_span: int) -> Tuple[pd.DataFrame, pd.Series]:
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


def mean_correlation(matrix: pd.DataFrame) -> float:
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
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def safe_mae(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(np.abs(values)))


def safe_rmse(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values))))


def compute_sector_dispersion(frame: pd.DataFrame) -> float:
    if "predicted_return" not in frame.columns or frame.empty:
        return 0.0
    sector_means = frame.groupby("sector")["predicted_return"].mean().replace([np.inf, -np.inf], np.nan)
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
