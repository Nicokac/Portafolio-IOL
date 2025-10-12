"""Shared cache helpers for market history and fundamentals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import pandas as pd

from services.cache.core import CacheService
from shared.settings import settings

DEFAULT_TTL_SECONDS = float(getattr(settings, "MARKET_DATA_CACHE_TTL", 6 * 60 * 60))
PREDICTION_TTL_SECONDS = 4 * 60 * 60


def _normalize_symbol(symbol: str | None) -> str:
    if symbol is None:
        return ""
    text = str(symbol).strip()
    return text.upper()


def _normalize_iterable(values: Iterable[str | None]) -> tuple[str, ...]:
    normalized = [_normalize_symbol(value) for value in values if _normalize_symbol(value)]
    return tuple(sorted(dict.fromkeys(normalized)))


def _clone_dataframe(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas.DataFrame instance for cached value")
    if df.empty:
        return df.copy()
    return df.copy(deep=True)


@dataclass
class MarketDataCache:
    """Manage cached market datasets with TTL invalidation."""

    history_cache: CacheService
    fundamentals_cache: CacheService
    prediction_cache: CacheService
    default_ttl: float = DEFAULT_TTL_SECONDS
    prediction_ttl: float = PREDICTION_TTL_SECONDS

    def __init__(
        self,
        *,
        history_cache: CacheService | None = None,
        fundamentals_cache: CacheService | None = None,
        prediction_cache: CacheService | None = None,
        default_ttl: float | None = None,
        prediction_ttl: float | None = None,
    ) -> None:
        self.history_cache = history_cache or CacheService(namespace="market_history")
        self.fundamentals_cache = fundamentals_cache or CacheService(
            namespace="market_fundamentals"
        )
        self.prediction_cache = prediction_cache or CacheService(
            namespace="market_predictions"
        )
        if default_ttl is not None:
            self.default_ttl = float(default_ttl)
        if prediction_ttl is not None:
            self.prediction_ttl = float(prediction_ttl)

    def _effective_ttl(self, ttl_seconds: float | None) -> float | None:
        if ttl_seconds is None:
            return self.default_ttl
        return float(ttl_seconds)

    def _effective_prediction_ttl(self, ttl_seconds: float | None) -> float | None:
        if ttl_seconds is None:
            return self.prediction_ttl
        return float(ttl_seconds)

    def _history_key(
        self,
        symbols: Sequence[str],
        *,
        period: str,
        benchmark: str | None = None,
        extra: Sequence[str] | None = None,
    ) -> str:
        normalized_symbols = _normalize_iterable(symbols)
        normalized_extra = _normalize_iterable(extra or [])
        benchmark_key = _normalize_symbol(benchmark)
        period_key = str(period or "").strip().lower()
        return "|".join(
            [
                "history",
                period_key or "-",
                benchmark_key or "-",
                ",".join(normalized_symbols) or "-",
                ",".join(normalized_extra) or "-",
            ]
        )

    def _fundamentals_key(
        self,
        symbols: Sequence[str],
        *,
        sectors: Sequence[str] | None = None,
    ) -> str:
        normalized_symbols = _normalize_iterable(symbols)
        sector_key = _normalize_iterable(sectors or [])
        return "|".join(
            [
                "fundamentals",
                ",".join(normalized_symbols) or "-",
                ",".join(sector_key) or "-",
            ]
        )

    def build_prediction_key(
        self,
        symbols: Sequence[str],
        *,
        span: int,
        sectors: Sequence[str] | None = None,
        period: str | None = None,
    ) -> str:
        normalized_symbols = _normalize_iterable(symbols)
        normalized_sectors = _normalize_iterable(sectors or [])
        span_key = str(int(span) if span is not None else 0)
        period_key = str(period or "").strip().lower() or "-"
        return "|".join(
            [
                "predictions",
                span_key,
                period_key,
                ",".join(normalized_symbols) or "-",
                ",".join(normalized_sectors) or "-",
            ]
        )

    def get_history(
        self,
        symbols: Sequence[str],
        *,
        loader: Callable[[], pd.DataFrame],
        period: str,
        benchmark: str | None = None,
        extra: Sequence[str] | None = None,
        ttl_seconds: float | None = None,
    ) -> pd.DataFrame:
        key = self._history_key(symbols, period=period, benchmark=benchmark, extra=extra)
        cached = self.history_cache.get(key)
        if isinstance(cached, pd.DataFrame):
            return _clone_dataframe(cached)  # type: ignore[return-value]
        value = loader()
        cloned = _clone_dataframe(value)
        if cloned is not None:
            self.history_cache.set(key, cloned, ttl=self._effective_ttl(ttl_seconds))
        return value

    def invalidate_history(
        self,
        symbols: Sequence[str],
        *,
        period: str,
        benchmark: str | None = None,
        extra: Sequence[str] | None = None,
    ) -> None:
        key = self._history_key(symbols, period=period, benchmark=benchmark, extra=extra)
        self.history_cache.invalidate(key)

    def get_fundamentals(
        self,
        symbols: Sequence[str],
        *,
        loader: Callable[[], pd.DataFrame],
        sectors: Sequence[str] | None = None,
        ttl_seconds: float | None = None,
    ) -> pd.DataFrame:
        key = self._fundamentals_key(symbols, sectors=sectors)
        cached = self.fundamentals_cache.get(key)
        if isinstance(cached, pd.DataFrame):
            return _clone_dataframe(cached)  # type: ignore[return-value]
        value = loader()
        cloned = _clone_dataframe(value)
        if cloned is not None:
            self.fundamentals_cache.set(key, cloned, ttl=self._effective_ttl(ttl_seconds))
        return value

    def invalidate_fundamentals(
        self,
        symbols: Sequence[str],
        *,
        sectors: Sequence[str] | None = None,
    ) -> None:
        key = self._fundamentals_key(symbols, sectors=sectors)
        self.fundamentals_cache.invalidate(key)

    def get_predictions(
        self,
        symbols: Sequence[str],
        *,
        loader: Callable[[], pd.DataFrame],
        span: int,
        sectors: Sequence[str] | None = None,
        period: str | None = None,
        ttl_seconds: float | None = None,
    ) -> tuple[pd.DataFrame, bool]:
        key = self.build_prediction_key(
            symbols,
            span=span,
            sectors=sectors,
            period=period,
        )
        cached = self.prediction_cache.get(key)
        if isinstance(cached, pd.DataFrame):
            return _clone_dataframe(cached) or pd.DataFrame(), True  # type: ignore[return-value]
        value = loader()
        cloned = _clone_dataframe(value)
        if cloned is not None:
            self.prediction_cache.set(
                key,
                cloned,
                ttl=self._effective_prediction_ttl(ttl_seconds),
            )
        return value, False

    def invalidate_predictions(
        self,
        symbols: Sequence[str],
        *,
        span: int,
        sectors: Sequence[str] | None = None,
        period: str | None = None,
    ) -> None:
        key = self.build_prediction_key(
            symbols,
            span=span,
            sectors=sectors,
            period=period,
        )
        self.prediction_cache.invalidate(key)


_default_cache = MarketDataCache()


def get_market_data_cache() -> MarketDataCache:
    """Return the shared market data cache instance."""

    return _default_cache


__all__ = [
    "MarketDataCache",
    "get_market_data_cache",
    "DEFAULT_TTL_SECONDS",
]
