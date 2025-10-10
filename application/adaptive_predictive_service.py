from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from application.backtesting_service import BacktestingService
from services.cache import CacheService

LOGGER = logging.getLogger(__name__)

_CACHE_NAMESPACE = "adaptive_predictive"
_STATE_KEY = "adaptive_state"
_CORR_KEY = "adaptive_correlations"
_TTL_SECONDS = 12 * 60 * 60  # 12 horas
_MAX_HISTORY_ROWS = 720
_DEFAULT_EMA_SPAN = 5

_CACHE = CacheService(namespace=_CACHE_NAMESPACE)


@dataclass(frozen=True)
class AdaptiveModelState:
    """Container for adaptive learning cached state."""

    history: pd.DataFrame
    last_updated: pd.Timestamp | None

    def copy(self) -> "AdaptiveModelState":
        return AdaptiveModelState(history=self.history.copy(), last_updated=self.last_updated)


def _ensure_timestamp(value: object) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value.tz_localize(None) if value.tzinfo else value
    if isinstance(value, str):
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except Exception:
            parsed = pd.NaT
        if pd.isna(parsed):
            return pd.Timestamp.utcnow().normalize()
        return parsed.tz_localize(None) if getattr(parsed, "tzinfo", None) else parsed
    if isinstance(value, (int, float)):
        try:
            parsed = pd.to_datetime(value, unit="s", errors="coerce")
        except Exception:
            parsed = pd.NaT
        if pd.isna(parsed):
            return pd.Timestamp.utcnow().normalize()
        return parsed
    return pd.Timestamp.utcnow().normalize()


def _initial_state() -> AdaptiveModelState:
    empty_history = pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "sector": pd.Series(dtype="string"),
            "normalized_error": pd.Series(dtype=float),
        }
    )
    return AdaptiveModelState(history=empty_history, last_updated=None)


def _normalise_predictions(frame: pd.DataFrame | None) -> pd.DataFrame:
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
        df["timestamp"] = df["timestamp"].apply(_ensure_timestamp)
    else:
        df["timestamp"] = pd.Timestamp.utcnow().normalize()
    columns = ["sector"]
    if "predicted_return" in df.columns:
        columns.append("predicted_return")
    elif "predicted_return_pct" in df.columns:
        df = df.rename(columns={"predicted_return_pct": "predicted_return"})
        columns.append("predicted_return")
    else:
        df["predicted_return"] = np.nan
        columns.append("predicted_return")
    columns.append("timestamp")
    result = df[columns]
    result["predicted_return"] = pd.to_numeric(
        result.get("predicted_return"), errors="coerce"
    ).astype(float)
    return result.dropna(subset=["sector"])


def _normalise_actuals(frame: pd.DataFrame | None) -> pd.DataFrame:
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
        df["timestamp"] = df["timestamp"].apply(_ensure_timestamp)
    else:
        df["timestamp"] = pd.Timestamp.utcnow().normalize()
    if "actual_return" not in df.columns and "realized_return" in df.columns:
        df = df.rename(columns={"realized_return": "actual_return"})
    df["actual_return"] = pd.to_numeric(df.get("actual_return"), errors="coerce").astype(float)
    return df[["sector", "actual_return", "timestamp"]]


def _merge_inputs(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    timestamp: pd.Timestamp | None,
) -> pd.DataFrame:
    if predictions.empty or actuals.empty:
        return pd.DataFrame(columns=["timestamp", "sector", "predicted_return", "actual_return"])
    merged = pd.merge(predictions, actuals, on=["sector", "timestamp"], how="outer")
    if timestamp is not None:
        merged["timestamp"] = _ensure_timestamp(timestamp)
    else:
        merged["timestamp"] = merged["timestamp"].apply(_ensure_timestamp)
    merged["predicted_return"] = pd.to_numeric(
        merged.get("predicted_return"), errors="coerce"
    ).astype(float)
    merged["actual_return"] = pd.to_numeric(
        merged.get("actual_return"), errors="coerce"
    ).astype(float)
    merged = merged.dropna(subset=["sector"])
    return merged


def _append_history(
    state: AdaptiveModelState,
    normalized_rows: pd.DataFrame,
) -> AdaptiveModelState:
    if normalized_rows.empty:
        return state
    history = state.history.copy()
    history = pd.concat([history, normalized_rows], ignore_index=True)
    history = history.sort_values("timestamp")
    if len(history) > _MAX_HISTORY_ROWS:
        history = history.iloc[-_MAX_HISTORY_ROWS :]
    last_timestamp = None
    if not normalized_rows.empty:
        last_timestamp = normalized_rows["timestamp"].iloc[-1]
    return AdaptiveModelState(history=history.reset_index(drop=True), last_updated=last_timestamp)


def _compute_beta_shift(pivot: pd.DataFrame, *, ema_span: int) -> tuple[pd.DataFrame, pd.Series]:
    if pivot.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    smoothed = pivot.sort_index().ewm(span=max(ema_span, 1), adjust=False).mean()
    last_row = smoothed.iloc[-1]
    beta_shift = -last_row.fillna(0.0)
    corr = smoothed.corr().fillna(0.0)
    if not corr.empty:
        corr.values[np.diag_indices_from(corr.values)] = 1.0
    return corr, beta_shift


def _prepare_normalized_frame(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame(columns=["timestamp", "sector", "normalized_error"])
    frame = merged.copy()
    frame = frame.dropna(subset=["sector", "predicted_return", "actual_return"])
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "sector", "normalized_error"])
    frame["timestamp"] = frame["timestamp"].apply(_ensure_timestamp)
    frame = frame.sort_values("timestamp")
    frame["error"] = frame["predicted_return"] - frame["actual_return"]
    denominator = frame["actual_return"].abs().clip(lower=1e-4)
    frame["normalized_error"] = (frame["error"] / denominator).clip(-5.0, 5.0)
    return frame[["timestamp", "sector", "normalized_error"]]


def _pivot_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    pivot = history.pivot_table(
        index="timestamp",
        columns="sector",
        values="normalized_error",
        aggfunc="mean",
    )
    return pivot.sort_index()


def _mean_correlation(matrix: pd.DataFrame) -> float:
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


def update_model(
    predictions: pd.DataFrame | None,
    actuals: pd.DataFrame | None,
    *,
    cache: CacheService | None = None,
    ema_span: int = _DEFAULT_EMA_SPAN,
    timestamp: pd.Timestamp | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Update the adaptive state using normalized prediction errors."""

    normalized_predictions = _normalise_predictions(predictions)
    normalized_actuals = _normalise_actuals(actuals)
    merged = _merge_inputs(normalized_predictions, normalized_actuals, timestamp=timestamp)
    normalized_frame = _prepare_normalized_frame(merged)

    active_cache = cache or _CACHE
    state = active_cache.get(_STATE_KEY)
    if not isinstance(state, AdaptiveModelState):
        state = _initial_state()

    updated_state = _append_history(state, normalized_frame)
    pivot = _pivot_history(updated_state.history)
    adaptive_corr, beta_shift = _compute_beta_shift(pivot, ema_span=max(int(ema_span), 1))

    result = {
        "timestamp": timestamp or (normalized_frame["timestamp"].max() if not normalized_frame.empty else None),
        "normalized": normalized_frame.copy(),
        "history": updated_state.history.copy(),
        "correlation_matrix": adaptive_corr.copy(),
        "beta_shift": beta_shift.copy(),
    }

    if persist:
        active_cache.set(_STATE_KEY, updated_state, ttl=_TTL_SECONDS)
        active_cache.set(_CORR_KEY, adaptive_corr.copy(), ttl=_TTL_SECONDS)

    return result


def _empty_history_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "sector", "predicted_return", "actual_return"])


def prepare_adaptive_history(
    opportunities: pd.DataFrame | None,
    *,
    backtesting_service: BacktestingService | None = None,
    span: int = _DEFAULT_EMA_SPAN,
    max_symbols: int = 12,
) -> pd.DataFrame:
    """Build a sector-level historical dataset using cached backtests."""

    if not isinstance(opportunities, pd.DataFrame) or opportunities.empty:
        return _empty_history_frame()

    svc = backtesting_service or BacktestingService()
    frame = opportunities.copy()
    if "symbol" not in frame.columns and "ticker" in frame.columns:
        frame = frame.rename(columns={"ticker": "symbol"})
    frame["symbol"] = (
        frame.get("symbol", pd.Series(dtype=str))
        .astype("string")
        .str.upper()
        .str.strip()
    )
    frame["sector"] = (
        frame.get("sector", pd.Series(dtype=str))
        .astype("string")
        .str.strip()
        .replace({"": "Sin sector"})
    )
    frame = frame.dropna(subset=["symbol", "sector"])
    frame = frame.drop_duplicates(subset=["symbol"])
    if frame.empty:
        return _empty_history_frame()

    rows: list[dict[str, Any]] = []
    selected = frame.head(max_symbols)
    for symbol, sector in zip(selected["symbol"], selected["sector"]):
        try:
            backtest = svc.run(str(symbol))
        except Exception:
            LOGGER.debug("Falló la preparación histórica para %s", symbol, exc_info=True)
            continue
        if backtest.empty:
            continue
        returns = pd.DataFrame({
            "predicted": pd.to_numeric(backtest.get("strategy_ret"), errors="coerce"),
            "actual": pd.to_numeric(backtest.get("ret"), errors="coerce"),
        }).dropna()
        if returns.empty:
            continue
        predicted_series = returns["predicted"].ewm(span=max(int(span), 1), adjust=False).mean() * 100.0
        actual_series = returns["actual"] * 100.0
        timestamps = pd.to_datetime(returns.index, errors="coerce")
        assembled = pd.DataFrame(
            {
                "timestamp": timestamps,
                "sector": str(sector),
                "symbol": str(symbol),
                "predicted_return": predicted_series,
                "actual_return": actual_series,
            }
        ).dropna(subset=["timestamp"])
        if assembled.empty:
            continue
        rows.append(assembled)

    if not rows:
        return _empty_history_frame()

    history = pd.concat(rows, ignore_index=True)
    history = history.groupby(["timestamp", "sector"], as_index=False)[
        ["predicted_return", "actual_return"]
    ].mean()
    history = history.sort_values("timestamp").reset_index(drop=True)
    return history


def generate_synthetic_history(recommendations: pd.DataFrame, periods: int = 6) -> pd.DataFrame:
    if not isinstance(recommendations, pd.DataFrame) or recommendations.empty:
        return _empty_history_frame()

    df = recommendations.copy()
    if "sector" not in df.columns:
        return _empty_history_frame()
    df["sector"] = df.get("sector", pd.Series(dtype=str)).astype("string").str.strip()
    df = df[df["sector"] != ""]
    if df.empty:
        return _empty_history_frame()

    if "predicted_return_pct" in df.columns:
        base_pred = pd.to_numeric(df["predicted_return_pct"], errors="coerce")
    elif "predicted_return" in df.columns:
        base_pred = pd.to_numeric(df["predicted_return"], errors="coerce")
    else:
        base_pred = pd.Series(np.nan, index=df.index)
    base_pred = base_pred.fillna(base_pred.mean()).fillna(3.0)

    rows: list[dict[str, Any]] = []
    sectors = df["sector"].unique().tolist()
    now = pd.Timestamp.utcnow().normalize()
    for idx, sector in enumerate(sectors):
        sector_bias = 0.6 + idx * 0.35
        sector_base = float(base_pred[df["sector"] == sector].mean())
        for step in range(periods):
            ts = now - pd.Timedelta(days=periods - step)
            seasonal = np.sin((step + 1) / (periods + 1) * np.pi) * 0.4
            predicted = sector_base + seasonal
            actual = predicted - sector_bias + ((step % 2) * 0.15)
            rows.append(
                {
                    "timestamp": ts,
                    "sector": sector,
                    "predicted_return": predicted,
                    "actual_return": actual,
                }
            )
    history = pd.DataFrame(rows)
    history = history.sort_values("timestamp").reset_index(drop=True)
    return history


def simulate_adaptive_forecast(
    history: pd.DataFrame | None,
    *,
    ema_span: int = _DEFAULT_EMA_SPAN,
    cache: CacheService | None = None,
    persist: bool = True,
    rolling_window: int = 20,
) -> dict[str, Any]:
    """Run an adaptive backtest and expose error metrics and correlations."""

    if not isinstance(history, pd.DataFrame) or history.empty:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "bias": 0.0,
            "raw_mae": 0.0,
            "raw_rmse": 0.0,
            "raw_bias": 0.0,
            "beta_shift": pd.Series(dtype=float),
            "correlation_matrix": pd.DataFrame(),
            "historical_correlation": pd.DataFrame(),
            "rolling_correlation": pd.DataFrame(),
            "summary": {},
            "steps": pd.DataFrame(),
        }

    df = history.copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "sector", "predicted_return", "actual_return"])
    if df.empty:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "bias": 0.0,
            "raw_mae": 0.0,
            "raw_rmse": 0.0,
            "raw_bias": 0.0,
            "beta_shift": pd.Series(dtype=float),
            "correlation_matrix": pd.DataFrame(),
            "historical_correlation": pd.DataFrame(),
            "rolling_correlation": pd.DataFrame(),
            "summary": {},
            "steps": pd.DataFrame(),
        }

    df = df.sort_values("timestamp")

    pivot_actual = df.pivot_table(
        index="timestamp",
        columns="sector",
        values="actual_return",
        aggfunc="mean",
    )
    historical_corr = pivot_actual.corr().fillna(0.0)
    if not historical_corr.empty:
        historical_corr.values[np.diag_indices_from(historical_corr.values)] = 1.0
    window = max(2, min(int(rolling_window), len(pivot_actual)))
    rolling_corr = pivot_actual.tail(window).corr().fillna(0.0)
    if not rolling_corr.empty:
        rolling_corr.values[np.diag_indices_from(rolling_corr.values)] = 1.0

    working_cache = cache or (_CACHE if persist else CacheService(namespace=f"{_CACHE_NAMESPACE}_sim"))

    grouped = list(df.groupby("timestamp"))
    raw_errors: list[float] = []
    adjusted_errors: list[float] = []
    step_rows: list[dict[str, Any]] = []
    last_result: dict[str, Any] | None = None

    for ts, group in grouped:
        predictions = group[["sector", "predicted_return"]]
        actuals = group[["sector", "actual_return"]]
        result = update_model(
            predictions,
            actuals,
            cache=working_cache,
            ema_span=ema_span,
            timestamp=ts,
            persist=persist,
        )
        beta_shift = result.get("beta_shift", pd.Series(dtype=float))
        last_result = result
        for row in group.itertuples(index=False):
            sector = getattr(row, "sector")
            predicted = float(getattr(row, "predicted_return"))
            actual = float(getattr(row, "actual_return"))
            raw_error = predicted - actual
            adjustment = float(beta_shift.get(sector, 0.0)) if isinstance(beta_shift, pd.Series) else 0.0
            factor = float(np.clip(1.0 + adjustment, 0.0, 2.0))
            adjusted_prediction = predicted * factor
            adj_error = adjusted_prediction - actual
            raw_errors.append(raw_error)
            adjusted_errors.append(adj_error)
            step_rows.append(
                {
                    "timestamp": ts,
                    "sector": sector,
                    "raw_prediction": predicted,
                    "adjusted_prediction": adjusted_prediction,
                    "actual_return": actual,
                    "beta_adjustment": adjustment,
                }
            )

    raw_errors_array = np.array(raw_errors, dtype=float)
    adjusted_errors_array = np.array(adjusted_errors, dtype=float)

    def _safe_mean(arr: np.ndarray) -> float:
        if arr.size == 0:
            return 0.0
        return float(np.mean(arr))

    def _safe_mae(arr: np.ndarray) -> float:
        if arr.size == 0:
            return 0.0
        return float(np.mean(np.abs(arr)))

    def _safe_rmse(arr: np.ndarray) -> float:
        if arr.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(arr))))

    mae = _safe_mae(adjusted_errors_array)
    raw_mae = _safe_mae(raw_errors_array)
    rmse = _safe_rmse(adjusted_errors_array)
    raw_rmse = _safe_rmse(raw_errors_array)
    bias = _safe_mean(adjusted_errors_array)
    raw_bias = _safe_mean(raw_errors_array)

    beta_shift = last_result.get("beta_shift", pd.Series(dtype=float)) if last_result else pd.Series(dtype=float)
    adaptive_corr = last_result.get("correlation_matrix", pd.DataFrame()) if last_result else pd.DataFrame()

    summary = {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "raw_mae": raw_mae,
        "raw_rmse": raw_rmse,
        "raw_bias": raw_bias,
        "beta_mean": float(beta_shift.mean()) if isinstance(beta_shift, pd.Series) and not beta_shift.empty else float("nan"),
        "correlation_mean": _mean_correlation(adaptive_corr),
        "sector_dispersion": float(beta_shift.std(ddof=0)) if isinstance(beta_shift, pd.Series) and beta_shift.size else float("nan"),
    }

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "raw_mae": raw_mae,
        "raw_rmse": raw_rmse,
        "raw_bias": raw_bias,
        "beta_shift": beta_shift.copy() if isinstance(beta_shift, pd.Series) else pd.Series(dtype=float),
        "correlation_matrix": adaptive_corr.copy() if isinstance(adaptive_corr, pd.DataFrame) else pd.DataFrame(),
        "historical_correlation": historical_corr.copy(),
        "rolling_correlation": rolling_corr.copy(),
        "summary": summary,
        "steps": pd.DataFrame(step_rows),
    }


__all__ = [
    "AdaptiveModelState",
    "generate_synthetic_history",
    "prepare_adaptive_history",
    "simulate_adaptive_forecast",
    "update_model",
]
