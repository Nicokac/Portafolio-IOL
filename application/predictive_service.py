from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from application.backtesting_service import BacktestingService
from services.cache import CacheService

LOGGER = logging.getLogger(__name__)

_CACHE_NAMESPACE = "predictive"
_CACHE_KEY = "sector_predictions"
_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 horas

_CACHE = CacheService(namespace=_CACHE_NAMESPACE)
_CACHE_HITS = 0
_CACHE_MISSES = 0


@dataclass(frozen=True)
class PredictiveSnapshot:
    """Container that surfaces statistics for cached predictions."""

    hits: int
    misses: int

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_ratio(self) -> float:
        total = self.total
        if total <= 0:
            return 0.0
        return float(self.hits) / float(total)


def _normalise_opportunities(opportunities: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(opportunities, pd.DataFrame) or opportunities.empty:
        return pd.DataFrame(columns=["symbol", "sector"])
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
        .fillna("Sin sector")
        .str.strip()
    )
    frame = frame[frame["symbol"] != ""]
    return frame[["symbol", "sector"]]


def _compute_symbol_prediction(
    backtesting_service: BacktestingService,
    symbol: str,
    *,
    span: int,
) -> tuple[pd.Series, float] | None:
    try:
        backtest = backtesting_service.run(symbol, strategy="sma")
    except Exception:  # pragma: no cover - defensive logging
        LOGGER.exception("Falló la ejecución de backtesting para %s", symbol)
        return None

    if backtest.empty or "strategy_ret" not in backtest.columns:
        return None

    returns = pd.to_numeric(backtest["strategy_ret"], errors="coerce").dropna()
    if returns.empty:
        return None

    ema = returns.ewm(span=max(span, 1), adjust=False).mean()
    predicted = float(ema.iloc[-1]) * 100.0
    return returns, predicted


def _aggregate_sector_predictions(
    frame: pd.DataFrame,
    *,
    backtesting_service: BacktestingService,
    span: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for sector, group in frame.groupby("sector"):
        symbol_returns: dict[str, pd.Series] = {}
        symbol_predictions: list[tuple[str, float]] = []

        for symbol in group["symbol"]:
            result = _compute_symbol_prediction(
                backtesting_service,
                str(symbol),
                span=span,
            )
            if result is None:
                continue
            returns, predicted = result
            symbol = str(symbol)
            symbol_returns[symbol] = returns
            symbol_predictions.append((symbol, predicted))

        if not symbol_predictions:
            continue

        if symbol_returns:
            aligned = pd.DataFrame(symbol_returns).dropna(how="all")
        else:
            aligned = pd.DataFrame()

        if aligned.shape[1] >= 2:
            corr_matrix = aligned.corr().replace([np.inf, -np.inf], np.nan)
            np.fill_diagonal(corr_matrix.values, np.nan)
            avg_corr = corr_matrix.mean(axis=1, skipna=True).fillna(0.0)
        else:
            symbols = [symbol for symbol, _ in symbol_predictions]
            avg_corr = pd.Series(0.0, index=symbols)

        weights: list[float] = []
        predictions: list[float] = []
        for symbol, predicted in symbol_predictions:
            correlation = float(avg_corr.get(symbol, 0.0))
            penalty = max(correlation, 0.0)
            weight = 1.0 / (1.0 + penalty)
            weights.append(weight)
            predictions.append(predicted)

        weights_array = np.array(weights, dtype=float)
        if not np.isfinite(weights_array).all() or weights_array.sum() <= 0:
            weights_array = np.ones_like(weights_array)
        weights_array = weights_array / weights_array.sum()

        predicted_sector = float(np.dot(weights_array, predictions))
        avg_corr_value = float(np.nanmean(avg_corr.to_numpy())) if not avg_corr.empty else 0.0
        confidence = float(max(0.0, min(1.0, 1.0 - max(avg_corr_value, 0.0))))

        rows.append(
            {
                "sector": str(sector),
                "predicted_return": predicted_sector,
                "sample_size": len(symbol_predictions),
                "avg_correlation": avg_corr_value,
                "confidence": confidence,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["sector", "predicted_return", "sample_size", "avg_correlation", "confidence"],
        )

    result = pd.DataFrame(rows)
    result = result.sort_values("predicted_return", ascending=False).reset_index(drop=True)
    return result


def predict_sector_performance(
    opportunities: pd.DataFrame | None,
    *,
    backtesting_service: BacktestingService | None = None,
    cache: CacheService | None = None,
    span: int = 10,
) -> pd.DataFrame:
    """Return expected sector returns using EMA-smoothed backtests."""

    global _CACHE_HITS, _CACHE_MISSES

    active_cache = cache or _CACHE
    cached = active_cache.get(_CACHE_KEY)
    if isinstance(cached, pd.DataFrame):
        _CACHE_HITS += 1
        return cached.copy()

    _CACHE_MISSES += 1

    frame = _normalise_opportunities(opportunities)
    if frame.empty:
        empty = pd.DataFrame(
            columns=["sector", "predicted_return", "sample_size", "avg_correlation", "confidence"],
        )
        active_cache.set(_CACHE_KEY, empty, ttl=_CACHE_TTL_SECONDS)
        return empty

    service = backtesting_service or BacktestingService()
    predictions = _aggregate_sector_predictions(
        frame,
        backtesting_service=service,
        span=span,
    )

    active_cache.set(_CACHE_KEY, predictions, ttl=_CACHE_TTL_SECONDS)
    return predictions.copy()


def get_cache_stats() -> PredictiveSnapshot:
    """Expose current cache counters for predictive workloads."""

    return PredictiveSnapshot(hits=_CACHE_HITS, misses=_CACHE_MISSES)


def reset_cache() -> None:
    """Utility mainly intended for tests to clear predictive caches."""

    global _CACHE_HITS, _CACHE_MISSES
    _CACHE.clear()
    _CACHE_HITS = 0
    _CACHE_MISSES = 0


__all__ = [
    "PredictiveSnapshot",
    "predict_sector_performance",
    "get_cache_stats",
    "reset_cache",
]
