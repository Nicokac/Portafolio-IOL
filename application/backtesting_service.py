"""Lightweight backtesting orchestration for the v0.5.0 transition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Protocol

import pandas as pd

from application.ta_service import run_backtest
from services.cache import CacheService
from shared.errors import AppError

__all__ = [
    "BacktestingService",
    "FixturePriceLoader",
    "load_prices_from_fixture",
]


class PriceLoader(Protocol):
    """Callable responsible for returning price data for a symbol."""

    def __call__(self, symbol: str) -> pd.DataFrame:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class FixturePriceLoader:
    """Loader that reads precomputed OHLC indicators from fixture files."""

    fixtures_root: Path = Path("docs/fixtures/default")

    def __call__(self, symbol: str) -> pd.DataFrame:
        return load_prices_from_fixture(symbol, root=self.fixtures_root)


def load_prices_from_fixture(symbol: str, *, root: Path | None = None) -> pd.DataFrame:
    """Load a CSV fixture containing technical indicators for ``symbol``."""

    base_path = Path(root) if root is not None else Path("docs/fixtures/default")
    file_name = f"prices_{symbol.upper()}.csv"
    csv_path = base_path / file_name
    if not csv_path.exists():
        raise FileNotFoundError(f"No fixture found for symbol {symbol!r} at {csv_path}")

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise AppError(f"El fixture de precios para {symbol} está vacío.")

    normalized = frame.copy()
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
        normalized = normalized.dropna(subset=["date"])
        if not normalized.empty:
            normalized = normalized.set_index("date")
            normalized.index.name = "Date"
    normalized.columns = [col.strip() for col in normalized.columns]
    required = {"Close", "SMA_FAST", "SMA_SLOW"}
    if not required.issubset(set(normalized.columns)):
        missing = ", ".join(sorted(required - set(normalized.columns)))
        raise AppError(
            f"El fixture de precios para {symbol} no contiene las columnas necesarias: {missing}."
        )
    return normalized


class BacktestingService:
    """Facade to execute cached backtests on historical indicator data."""

    def __init__(
        self,
        *,
        cache: CacheService | None = None,
        data_loader: PriceLoader | None = None,
        ttl_seconds: float | None = 900.0,
    ) -> None:
        self._cache = cache or CacheService(namespace="backtesting")
        self._loader = data_loader or FixturePriceLoader()
        self._ttl_seconds = float(ttl_seconds) if ttl_seconds is not None else None
        self._lock = Lock()

    def load_prices(self, symbol: str) -> pd.DataFrame:
        """Load and cache indicator-enriched prices for ``symbol``."""

        key = f"prices::{symbol.upper()}"
        with self._lock:
            return self._cache.get_or_set(
                key,
                lambda: self._loader(symbol.upper()),
                ttl=self._ttl_seconds,
            )

    def run(self, symbol: str, *, strategy: str = "sma") -> pd.DataFrame:
        """Execute a lightweight backtest for ``symbol`` using ``strategy``."""

        prices = self.load_prices(symbol)
        result = run_backtest(prices, strategy=strategy)
        if result.empty:
            raise AppError(
                "No se pudo ejecutar el backtesting con los datos disponibles para la estrategia seleccionada."
            )
        return result

    def invalidate(self, symbol: str | None = None) -> None:
        """Invalidate cached prices for ``symbol`` or clear the full namespace."""

        if symbol is None:
            self._cache.clear()
            return
        self._cache.invalidate(f"prices::{symbol.upper()}")
