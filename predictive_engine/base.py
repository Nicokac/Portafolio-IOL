"""Core predictive routines extracted from the application layer."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable

import numpy as np
import pandas as pd

from predictive_engine import utils
from predictive_engine.models import (
    AdaptiveForecastResult,
    AdaptiveState,
    AdaptiveUpdateResult,
    CorrelationBundle,
    ModelMetrics,
    SectorPrediction,
    SectorPredictionSet,
)
from predictive_engine.storage import save_forecast_history


def compute_sector_predictions(
    opportunities: pd.DataFrame | None,
    *,
    backtesting_service: object,
    run_backtest: Callable[[object, str], pd.DataFrame | None],
    extract_series: Callable[[pd.DataFrame, str], pd.Series],
    ema_predictor: Callable[[pd.Series, int], float | None],
    average_correlation: Callable[[Dict[str, pd.Series]], pd.Series],
    span: int,
    logger: object | None = None,
) -> SectorPredictionSet:
    """Generate sector predictions using EMA-smoothed backtests."""

    if not isinstance(opportunities, pd.DataFrame) or opportunities.empty:
        return SectorPredictionSet()

    frame = opportunities.copy()
    required = [column for column in ["symbol", "sector"] if column in frame.columns]
    if len(required) < 2:
        return SectorPredictionSet()
    frame = frame.loc[:, required].dropna()
    if frame.empty:
        return SectorPredictionSet()

    rows: list[SectorPrediction] = []

    for sector, group in frame.groupby("sector"):
        symbol_returns: dict[str, pd.Series] = {}
        symbol_predictions: list[tuple[str, float]] = []

        for symbol in group["symbol"]:
            symbol_str = str(symbol)
            backtest = run_backtest(backtesting_service, symbol_str)
            if backtest is None:
                continue
            returns = extract_series(backtest, "strategy_ret")
            if returns.empty:
                continue
            predicted = ema_predictor(returns, span)
            if predicted is None:
                continue
            symbol_returns[symbol_str] = returns
            symbol_predictions.append((symbol_str, predicted))

        if not symbol_predictions:
            continue

        avg_corr = average_correlation(symbol_returns)
        if avg_corr.empty:
            avg_corr = pd.Series(
                0.0,
                index=[symbol for symbol, _ in symbol_predictions],
                dtype=float,
            )

        weights: list[float] = []
        predictions: list[float] = []
        for symbol, predicted in symbol_predictions:
            if isinstance(avg_corr, pd.Series):
                correlation = float(avg_corr.get(symbol, 0.0))
            else:
                correlation = 0.0
            penalty = max(correlation, 0.0)
            weight = 1.0 / (1.0 + penalty)
            weights.append(weight)
            predictions.append(predicted)

        weights_array = np.array(weights, dtype=float)
        if not np.isfinite(weights_array).all() or weights_array.sum() <= 0:
            weights_array = np.ones_like(weights_array)
        weights_array = weights_array / weights_array.sum()

        predicted_sector = float(np.dot(weights_array, predictions))
        if avg_corr.empty:
            avg_corr_value = 0.0
        else:
            avg_corr_value = float(np.nanmean(avg_corr.to_numpy()))
        confidence = float(max(0.0, min(1.0, 1.0 - max(avg_corr_value, 0.0))))

        rows.append(
            SectorPrediction(
                sector=str(sector),
                predicted_return=predicted_sector,
                sample_size=len(symbol_predictions),
                avg_correlation=avg_corr_value,
                confidence=confidence,
            )
        )

    return SectorPredictionSet(rows=rows)


def update_adaptive_state(
    predictions: pd.DataFrame | None,
    actuals: pd.DataFrame | None,
    *,
    state: AdaptiveState,
    ema_span: int,
    timestamp: pd.Timestamp | None,
    max_history_rows: int,
) -> AdaptiveUpdateResult:
    normalized_predictions = utils.normalise_predictions(predictions)
    normalized_actuals = utils.normalise_actuals(actuals)
    merged = utils.merge_inputs(
        normalized_predictions,
        normalized_actuals,
        timestamp=timestamp,
    )
    normalized_frame = utils.prepare_normalized_frame(merged)

    history, last_timestamp = utils.append_history(
        state.history,
        normalized_frame,
        max_rows=max_history_rows,
    )
    pivot = utils.pivot_history(history)
    corr_matrix, beta_shift = utils.compute_beta_shift(pivot, ema_span=ema_span)

    updated_state = AdaptiveState(
        history=history,
        last_updated=last_timestamp or state.last_updated,
    )
    return AdaptiveUpdateResult(
        state=updated_state,
        normalized=normalized_frame,
        correlation_matrix=corr_matrix,
        beta_shift=beta_shift,
        timestamp=timestamp or last_timestamp,
    )


def evaluate_model_metrics(
    adjusted_errors: Iterable[float],
    raw_errors: Iterable[float],
    *,
    beta_shift_avg: float,
    correlation_matrix: pd.DataFrame,
    sector_dispersion: float,
) -> ModelMetrics:
    adjusted_array = np.array(list(adjusted_errors), dtype=float)
    raw_array = np.array(list(raw_errors), dtype=float)
    mae = utils.safe_mae(adjusted_array)
    rmse = utils.safe_rmse(adjusted_array)
    bias = utils.safe_mean(adjusted_array)
    raw_mae = utils.safe_mae(raw_array)
    raw_rmse = utils.safe_rmse(raw_array)
    raw_bias = utils.safe_mean(raw_array)
    corr_mean = utils.mean_correlation(correlation_matrix)
    return ModelMetrics(
        mae=mae,
        rmse=rmse,
        bias=bias,
        raw_mae=raw_mae,
        raw_rmse=raw_rmse,
        raw_bias=raw_bias,
        beta_shift_avg=float(beta_shift_avg),
        correlation_mean=float(corr_mean),
        sector_dispersion=float(sector_dispersion),
    )


def _summarise_metrics(metrics: ModelMetrics) -> Dict[str, float | str]:
    def _fmt_percent(value: float) -> str:
        if not np.isfinite(value):
            return "-"
        return f"{value:.2f}%"

    def _fmt_float(value: float) -> str:
        if not np.isfinite(value):
            return "-"
        return f"{value:.2f}"

    summary: dict[str, float | str] = {
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "bias": metrics.bias,
        "raw_mae": metrics.raw_mae,
        "raw_rmse": metrics.raw_rmse,
        "raw_bias": metrics.raw_bias,
        "beta_shift_avg": metrics.beta_shift_avg,
        "sector_dispersion": metrics.sector_dispersion,
        "correlation_mean": metrics.correlation_mean,
    }
    summary["text"] = " | ".join(
        [
            f"MAE adaptativo: {_fmt_percent(metrics.mae)}",
            f"RMSE: {_fmt_percent(metrics.rmse)}",
            f"Bias: {_fmt_percent(metrics.bias)}",
            f"β-shift promedio: {_fmt_float(metrics.beta_shift_avg)}",
            f"σ sectorial: {_fmt_percent(metrics.sector_dispersion)}",
        ]
    )
    return summary


def calculate_adaptive_forecast(
    history: pd.DataFrame,
    *,
    ema_span: int,
    rolling_window: int,
    model_updater: Callable[
        [pd.DataFrame, pd.DataFrame, pd.Timestamp],
        AdaptiveUpdateResult,
    ],
    cache_metadata: Dict[str, float | str] | None = None,
    persist_history: bool = False,
    history_path: str | Path | None = "./data/forecast_history.parquet",
    history_saver: Callable[[pd.DataFrame, str | Path], object] | None = None,
) -> AdaptiveForecastResult:
    if not isinstance(history, pd.DataFrame) or history.empty:
        metrics = ModelMetrics(
            mae=0.0,
            rmse=0.0,
            bias=0.0,
            raw_mae=0.0,
            raw_rmse=0.0,
            raw_bias=0.0,
            beta_shift_avg=0.0,
            correlation_mean=float("nan"),
            sector_dispersion=0.0,
        )
        correlations = CorrelationBundle()
        summary = _summarise_metrics(metrics)
        return AdaptiveForecastResult(
            metrics=metrics,
            beta_shift=pd.Series(dtype=float),
            correlations=correlations,
            steps=pd.DataFrame(),
            cache_metadata=cache_metadata or {"hit_ratio": 0.0, "last_updated": "-"},
            summary=summary,
        )

    df = history.copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "sector", "predicted_return", "actual_return"])
    if df.empty:
        return calculate_adaptive_forecast(
            pd.DataFrame(),
            ema_span=ema_span,
            rolling_window=rolling_window,
            model_updater=model_updater,
            cache_metadata=cache_metadata,
        )

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

    grouped = list(df.groupby("timestamp"))
    raw_error_blocks: list[np.ndarray] = []
    adjusted_error_blocks: list[np.ndarray] = []
    step_frames: list[pd.DataFrame] = []
    beta_snapshots: list[pd.Series] = []
    last_update: AdaptiveUpdateResult | None = None

    for ts, group in grouped:
        predictions = group[["sector", "predicted_return"]]
        actuals = group[["sector", "actual_return"]]
        update = model_updater(predictions, actuals, ts)
        last_update = update
        beta_shift = update.beta_shift.copy()
        beta_shift.name = ts
        beta_snapshots.append(beta_shift)

        sector_series = group["sector"].astype("string")
        predicted_values = pd.to_numeric(
            group["predicted_return"],
            errors="coerce",
        ).to_numpy(dtype=float)
        actual_values = pd.to_numeric(
            group["actual_return"],
            errors="coerce",
        ).to_numpy(dtype=float)
        adjustments = (
            sector_series.map(update.beta_shift).fillna(0.0).to_numpy(dtype=float)
        )
        factors = np.clip(1.0 + adjustments, 0.0, 2.0)
        adjusted_predictions = predicted_values * factors

        raw_error_blocks.append(predicted_values - actual_values)
        adjusted_error_blocks.append(adjusted_predictions - actual_values)

        step_frames.append(
            pd.DataFrame(
                {
                    "timestamp": ts,
                    "sector": sector_series.to_numpy(),
                    "raw_prediction": predicted_values,
                    "adjusted_prediction": adjusted_predictions,
                    "actual_return": actual_values,
                    "beta_adjustment": adjustments,
                }
            )
        )

    if beta_snapshots:
        beta_frame = pd.DataFrame(beta_snapshots).sort_index()
        beta_diffs = beta_frame.fillna(0.0).diff().abs().dropna(how="all")
        if beta_diffs.empty:
            beta_shift_avg = 0.0
        else:
            beta_magnitudes = beta_diffs.mean(axis=1, skipna=True).dropna()
            if beta_magnitudes.empty:
                beta_shift_avg = 0.0
            else:
                beta_shift_avg = float(beta_magnitudes.mean())
    else:
        beta_shift_avg = 0.0

    last_beta = last_update.beta_shift if last_update else pd.Series(dtype=float)
    correlation_matrix = (
        last_update.correlation_matrix if last_update else pd.DataFrame()
    )
    sector_dispersion = utils.compute_sector_dispersion(df)
    raw_errors = (
        np.concatenate(raw_error_blocks)
        if raw_error_blocks
        else np.empty(0, dtype=float)
    )
    adjusted_errors = (
        np.concatenate(adjusted_error_blocks)
        if adjusted_error_blocks
        else np.empty(0, dtype=float)
    )

    metrics = evaluate_model_metrics(
        adjusted_errors,
        raw_errors,
        beta_shift_avg=beta_shift_avg,
        correlation_matrix=correlation_matrix,
        sector_dispersion=sector_dispersion,
    )
    summary = _summarise_metrics(metrics)

    correlations = CorrelationBundle(
        correlation_matrix=correlation_matrix.copy(),
        historical_correlation=historical_corr.copy(),
        rolling_correlation=rolling_corr.copy(),
    )

    cache_payload = cache_metadata or {"hit_ratio": 0.0, "last_updated": "-"}

    steps_frame = (
        pd.concat(step_frames, ignore_index=True)
        if step_frames
        else pd.DataFrame(
            columns=[
                "timestamp",
                "sector",
                "raw_prediction",
                "adjusted_prediction",
                "actual_return",
                "beta_adjustment",
            ]
        )
    )

    result = AdaptiveForecastResult(
        metrics=metrics,
        beta_shift=last_beta.copy(),
        correlations=correlations,
        steps=steps_frame,
        cache_metadata=cache_payload,
        summary=summary,
    )

    if persist_history and history_path and last_update is not None:
        saver = history_saver or save_forecast_history
        try:
            saver(last_update.state.history.copy(), history_path)
        except Exception:
            pass

    return result
