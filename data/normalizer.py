"""Shared normalisation helpers for predictive datasets."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

_DEFAULT_PREDICTION_COLUMNS = ["sector", "predicted_return", "timestamp"]
_DEFAULT_ACTUAL_COLUMNS = ["sector", "actual_return", "timestamp"]


def ensure_timestamp(
    value: object,
    *,
    default: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Coerce *value* into a timezone naive ``Timestamp``."""

    fallback = default or pd.Timestamp.utcnow().normalize()
    if isinstance(value, pd.Timestamp):
        return value.tz_localize(None) if value.tzinfo else value
    if isinstance(value, datetime):
        ts = pd.Timestamp(value)
        return ts.tz_localize(None) if ts.tzinfo else ts
    if isinstance(value, str):
        parsed = pd.to_datetime(value, errors="coerce")
        if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed):
            return parsed.tz_localize(None) if parsed.tzinfo else parsed
        return fallback
    if isinstance(value, (int, float)):
        parsed = pd.to_datetime(value, unit="s", errors="coerce")
        if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed):
            return parsed.tz_localize(None) if parsed.tzinfo else parsed
        return fallback
    return fallback


def _empty_predictions_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_DEFAULT_PREDICTION_COLUMNS)


def _empty_actuals_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_DEFAULT_ACTUAL_COLUMNS)


def normalise_predictions(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Normalise prediction records ensuring required columns exist."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return _empty_predictions_frame()
    df = frame.copy()
    if "sector" not in df.columns:
        return _empty_predictions_frame()
    df["sector"] = df.get("sector", pd.Series(dtype=str)).astype("string").str.strip().replace({"": "Sin sector"})
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(ensure_timestamp)
    else:
        df["timestamp"] = pd.Timestamp.utcnow().normalize()
    if "predicted_return" not in df.columns and "predicted_return_pct" in df.columns:
        df = df.rename(columns={"predicted_return_pct": "predicted_return"})
    if "predicted_return" not in df.columns:
        df["predicted_return"] = np.nan
    df["predicted_return"] = pd.to_numeric(
        df.get("predicted_return"),
        errors="coerce",
    ).astype(float)
    return df[_DEFAULT_PREDICTION_COLUMNS]


def normalise_actuals(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Normalise realised return records."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return _empty_actuals_frame()
    df = frame.copy()
    if "sector" not in df.columns:
        return _empty_actuals_frame()
    df["sector"] = df.get("sector", pd.Series(dtype=str)).astype("string").str.strip().replace({"": "Sin sector"})
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(ensure_timestamp)
    else:
        df["timestamp"] = pd.Timestamp.utcnow().normalize()
    if "actual_return" not in df.columns and "realized_return" in df.columns:
        df = df.rename(columns={"realized_return": "actual_return"})
    df["actual_return"] = pd.to_numeric(
        df.get("actual_return"),
        errors="coerce",
    ).astype(float)
    return df[_DEFAULT_ACTUAL_COLUMNS]


def merge_inputs(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    timestamp: pd.Timestamp | None,
) -> pd.DataFrame:
    """Merge prediction and actual frames with optional timestamp override."""

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
    """Compute per-sector normalised errors from merged inputs."""

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
