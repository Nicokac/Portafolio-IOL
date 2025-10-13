"""Sector-level predictive analytics with adaptive caching."""

from __future__ import annotations

import logging
from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import pandas as pd
import numpy as np

from application.backtesting_service import BacktestingService
from application.predictive_core import (
    PredictiveCacheState,
    average_correlation,
    compute_ema_prediction,
    extract_backtest_series,
    normalise_symbol_sector,
    run_backtest,
)
from domain.adaptive_cache_lock import adaptive_cache_lock
from services.cache import CacheService
from services.performance_metrics import track_function
from services.performance_timer import performance_timer
from shared.settings import PREDICTIVE_TTL_HOURS

from predictive_engine import __version__ as ENGINE_VERSION
from predictive_engine.adapters import build_sector_prediction_frame


LOGGER = logging.getLogger(__name__)

_CACHE_NAMESPACE = "predictive"
_CACHE_KEY_PREFIX = "sector_predictions"
_HISTORY_CACHE_PREFIX = "adaptive_history"
class FallbackCache:
    """Minimal cache stand-in used when the market data cache is unavailable."""

    hits: int
    misses: int
    last_updated_human: str

    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0
        self.last_updated_human = "-"

    def get(self, *_, **__) -> None:  # pragma: no cover - trivial behaviour
        return None

    def set(self, *_, **__) -> None:  # pragma: no cover - trivial behaviour
        return None

    def clear(self) -> None:  # pragma: no cover - trivial behaviour
        return None

    def set_ttl_override(self, *_: Any, **__: Any) -> None:  # pragma: no cover
        return None

    def get_effective_ttl(self) -> None:  # pragma: no cover - fallback has no TTL
        return None

    def remaining_ttl(self) -> None:  # pragma: no cover - fallback has no TTL
        return None

    def status(self) -> dict[str, Any]:
        return {"backend": "fallback", "available": False}


class FallbackMarketDataCache:
    """Fallback market data cache container for degraded mode."""

    def __init__(self) -> None:
        self.prediction_cache = FallbackCache()

    def build_prediction_key(
        self,
        symbols: list[str] | tuple[str, ...] | None = None,
        *,
        span: int | None = None,
        sectors: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        joined_symbols = ",".join(map(str, symbols or ()))
        joined_sectors = ",".join(map(str, sectors or ()))
        return f"fallback:{joined_symbols}:{joined_sectors}:{span}"

    def status(self) -> dict[str, Any]:
        return self.prediction_cache.status()


def lazy_get_cache() -> Any:
    """Return the market data cache or a safe fallback when unavailable."""

    try:
        from services.cache.market_data_cache import get_market_data_cache
    except ImportError:
        LOGGER.warning("⚠️ MarketDataCache unavailable — running with fallback cache")
        return FallbackMarketDataCache()

    try:
        return get_market_data_cache()
    except Exception:  # pragma: no cover - defensive degradation
        LOGGER.exception("No se pudo inicializar MarketDataCache, usando fallback")
        LOGGER.warning("⚠️ MarketDataCache unavailable — running with fallback cache")
        return FallbackMarketDataCache()


_MARKET_CACHE = lazy_get_cache()
_CACHE = getattr(_MARKET_CACHE, "prediction_cache", _MARKET_CACHE)
if hasattr(_CACHE, "set_ttl_override"):
    _CACHE.set_ttl_override(PREDICTIVE_TTL_HOURS * 3600.0)
_CACHE_STATE = PredictiveCacheState()

_ADAPTIVE_DEFAULT_SPAN = 5
_SYNTHETIC_DEFAULT_PERIODS = 6


def _cache_last_updated(cache: CacheService) -> str | None:
    try:
        value = getattr(cache, "last_updated_human", "-")
    except Exception:  # pragma: no cover - defensive
        return None
    if not value or value == "-":
        return None
    return str(value)


@dataclass(frozen=True)
class PredictiveSnapshot:
    """Container that surfaces statistics for cached predictions."""

    hits: int
    misses: int
    last_updated: str = "-"
    ttl_hours: float | None = None
    remaining_ttl: float | None = None

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_ratio(self) -> float:
        """Return the percentage of cache hits over total lookups."""

        total = self.total
        if total <= 0:
            return 0.0
        return float(self.hits) / float(total)

    def as_dict(self) -> dict[str, float | int | str]:
        """Expose snapshot information as a serialisable mapping."""

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hit_ratio,
            "last_updated": self.last_updated,
            "ttl_hours": self.ttl_hours,
            "remaining_ttl": self.remaining_ttl,
        }


def _prepare_opportunities(opportunities: pd.DataFrame | None) -> pd.DataFrame:
    frame = normalise_symbol_sector(opportunities)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=["symbol", "sector"])
    required = [column for column in ["symbol", "sector"] if column in frame.columns]
    if len(required) < 2:
        return pd.DataFrame(columns=["symbol", "sector"])
    return frame.loc[:, required].copy()


def _empty_history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["timestamp", "sector", "predicted_return", "actual_return"]
    )


def _build_history_cache_key(
    mode: str,
    symbols: list[str],
    sectors: list[str],
    span: int,
    max_symbols: int,
    periods: int,
) -> str:
    payload = "|".join(f"{sym}:{sec}" for sym, sec in zip(symbols, sectors))
    digest = hashlib.sha1(  # noqa: S324 - deterministic cache key
        payload.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    return (
        f"{_HISTORY_CACHE_PREFIX}:{mode}:{span}:{max_symbols}:{periods}:{digest}"
    )


def build_adaptive_history(
    data: pd.DataFrame | None,
    *,
    mode: str = "real",
    backtesting_service: BacktestingService | None = None,
    span: int = _ADAPTIVE_DEFAULT_SPAN,
    max_symbols: int = 12,
    periods: int = _SYNTHETIC_DEFAULT_PERIODS,
    cache: CacheService | None = None,
    ttl_hours: float | None = None,
    context: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Assemble an adaptive history frame either from backtests or synthetic data."""

    mode_key = str(mode or "real").strip().lower()
    if mode_key not in {"real", "synthetic"}:
        raise ValueError(
            "Modo de histórico adaptativo no soportado: solo se permite 'real' o 'synthetic'"
        )

    frame = normalise_symbol_sector(data)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return _empty_history_frame()

    working = frame.loc[:, [col for col in frame.columns if col in {"symbol", "sector"}]]
    if {"symbol", "sector"} - set(working.columns):
        return _empty_history_frame()

    working["symbol"] = (
        working.get("symbol", pd.Series(dtype=str))
        .astype("string")
        .fillna("")
        .str.upper()
        .str.strip()
    )
    working["sector"] = (
        working.get("sector", pd.Series(dtype=str))
        .astype("string")
        .fillna("")
        .str.strip()
    )
    working = working.dropna(subset=["symbol", "sector"])
    working = working[working["symbol"] != ""]
    if working.empty:
        return _empty_history_frame()

    selected = working.drop_duplicates(subset=["symbol"]).head(max(int(max_symbols), 1))
    symbols = selected["symbol"].tolist()
    sectors = selected["sector"].tolist()

    effective_ttl_hours = (
        float(ttl_hours)
        if ttl_hours is not None
        else float(PREDICTIVE_TTL_HOURS)
    )
    ttl_seconds = max(effective_ttl_hours, 0.0) * 3600.0

    active_cache = cache if cache is not None else _CACHE
    if isinstance(active_cache, CacheService):
        active_cache.set_ttl_override(ttl_seconds)

    cache_key = _build_history_cache_key(
        mode_key,
        symbols,
        sectors,
        span,
        max_symbols,
        periods,
    )

    with adaptive_cache_lock:
        cached = active_cache.get(cache_key) if active_cache is not None else None
        if isinstance(cached, pd.DataFrame):
            LOGGER.debug(
                "Histórico adaptativo recuperado desde caché (modo=%s, símbolos=%s)",
                mode_key,
                symbols,
            )
            return cached.copy()

    history = _empty_history_frame()
    if mode_key == "real":
        service = backtesting_service or BacktestingService()
        rows: list[pd.DataFrame] = []
        ema_span = max(int(span), 1)
        for symbol, sector in zip(symbols, sectors):
            try:
                backtest = run_backtest(service, str(symbol), logger=LOGGER)
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.exception(
                    "No se pudo obtener backtest para el histórico adaptativo",
                    extra={"symbol": symbol, "sector": sector},
                )
                continue
            if backtest is None:
                LOGGER.warning(
                    "Backtest inexistente para histórico adaptativo",
                    extra={"symbol": symbol, "sector": sector},
                )
                continue

            predicted_returns = extract_backtest_series(backtest, "strategy_ret")
            actual_returns = extract_backtest_series(backtest, "ret")
            aligned = pd.DataFrame(
                {
                    "predicted": predicted_returns,
                    "actual": actual_returns,
                }
            ).dropna()
            if aligned.empty:
                continue

            predicted_series = aligned["predicted"].ewm(
                span=ema_span,
                adjust=False,
            ).mean()
            actual_series = aligned["actual"]
            timestamps = pd.to_datetime(aligned.index, errors="coerce")
            assembled = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "sector": str(sector),
                    "predicted_return": predicted_series * 100.0,
                    "actual_return": actual_series * 100.0,
                }
            ).dropna(subset=["timestamp"])
            if assembled.empty:
                continue
            rows.append(assembled)

        if rows:
            history = pd.concat(rows, ignore_index=True)
            history = history.groupby(["timestamp", "sector"], as_index=False)[
                ["predicted_return", "actual_return"]
            ].mean()
            history = history.sort_values("timestamp").reset_index(drop=True)
    else:
        df = frame.copy()
        if "symbol" not in df.columns and "ticker" in df.columns:
            df = df.rename(columns={"ticker": "symbol"})
        df["symbol"] = (
            df.get("symbol", pd.Series(dtype=str))
            .astype("string")
            .fillna("")
            .str.upper()
            .str.strip()
        )
        df["sector"] = (
            df.get("sector", pd.Series(dtype=str))
            .astype("string")
            .fillna("")
            .str.strip()
        )
        df = df.dropna(subset=["symbol", "sector"])
        df = df[df["symbol"] != ""]
        if df.empty:
            history = _empty_history_frame()
        else:
            predicted_pct = pd.to_numeric(
                df.get("predicted_return_pct"), errors="coerce"
            )
            predicted_alt = pd.to_numeric(
                df.get("predicted_return"), errors="coerce"
            )
            if predicted_pct.isna().all():
                predicted_pct = predicted_alt
            if predicted_pct.isna().all():
                predicted_pct = pd.Series(np.nan, index=df.index)

            needs_scaling = predicted_pct.abs() > 1.0
            if needs_scaling.any():
                predicted_pct.loc[needs_scaling] = predicted_pct.loc[needs_scaling] / 100.0

            predicted_pct = predicted_pct.fillna(predicted_pct.mean())
            predicted_pct = predicted_pct.fillna(0.03)

            clipped = predicted_pct.clip(lower=-0.5, upper=0.5)
            if not clipped.equals(predicted_pct):
                profile = None
                if isinstance(context, Mapping):
                    profile = context.get("profile")
                for idx, original in predicted_pct.items():
                    clipped_value = clipped.loc[idx]
                    if not np.isfinite(original) or np.isclose(original, clipped_value):
                        continue
                    symbol = str(df.loc[idx, "symbol"]) if idx in df.index else ""
                    sector = str(df.loc[idx, "sector"]) if idx in df.index else ""
                    LOGGER.warning(
                        "Predicted return fuera de rango, truncando valor",
                        extra={
                            "symbol": symbol,
                            "sector": sector,
                            "profile": profile,
                            "valor_original": float(original),
                            "valor_truncado": float(clipped_value),
                            "modo": mode_key,
                        },
                    )
            predicted_pct = clipped

            rows = []
            now = pd.Timestamp.utcnow().normalize()
            for idx, sector in enumerate(df["sector"].unique().tolist()):
                sector_mask = df["sector"] == sector
                sector_base = float(predicted_pct.loc[sector_mask].mean())
                sector_bias = 0.006 + idx * 0.0035
                for step in range(int(max(periods, 1))):
                    ts = now - pd.Timedelta(days=int(periods) - step)
                    seasonal = float(
                        np.sin((step + 1) / (int(periods) + 1) * np.pi) * 0.04
                    )
                    predicted_decimal = sector_base + seasonal
                    actual_decimal = predicted_decimal - sector_bias + ((step % 2) * 0.015)
                    rows.append(
                        {
                            "timestamp": ts,
                            "sector": str(sector),
                            "predicted_return": predicted_decimal * 100.0,
                            "actual_return": actual_decimal * 100.0,
                        }
                    )
            if rows:
                history = pd.DataFrame(rows)
                history = history.sort_values("timestamp").reset_index(drop=True)

    if history.empty:
        return history

    with adaptive_cache_lock:
        if active_cache is not None:
            active_cache.set(cache_key, history, ttl=ttl_seconds)
            LOGGER.debug(
                "Histórico adaptativo almacenado en caché (modo=%s, símbolos=%s)",
                mode_key,
                symbols,
            )

    return history.copy()


@track_function("predict_sector_performance")
def predict_sector_performance(
    opportunities: pd.DataFrame | None,
    *,
    backtesting_service: BacktestingService | None = None,
    cache: CacheService | None = None,
    span: int = 10,
    ttl_hours: float | None = None,
) -> pd.DataFrame:
    """Return expected sector returns using EMA-smoothed backtests."""

    effective_ttl_hours = (
        float(ttl_hours)
        if ttl_hours is not None
        else float(PREDICTIVE_TTL_HOURS)
    )
    ttl_seconds = max(effective_ttl_hours, 0.0) * 3600.0
    active_cache: CacheService | None = cache if cache is not None else _CACHE
    if isinstance(active_cache, CacheService):
        active_cache.set_ttl_override(ttl_seconds)
    telemetry: dict[str, object] = {
        "status": "success",
        "span": int(span),
        "ttl_hours": float(effective_ttl_hours),
    }
    frame = _prepare_opportunities(opportunities)
    telemetry["opportunities"] = int(len(frame))
    symbols = (
        frame.get("symbol", pd.Series(dtype=str))
        .astype("string")
        .str.upper()
        .str.strip()
        .tolist()
    )
    sectors = (
        frame.get("sector", pd.Series(dtype=str))
        .astype("string")
        .str.strip()
        .tolist()
    )
    cache_key = f"{_CACHE_KEY_PREFIX}:{_MARKET_CACHE.build_prediction_key(symbols, span=span, sectors=sectors)}"

    LOGGER.debug(
        "Solicitando lock adaptativo para predicciones (span=%s, símbolos=%s, sectores=%s)",
        span,
        symbols,
        sectors,
    )

    cached_frame: pd.DataFrame | None = None
    cache_hit = False

    with adaptive_cache_lock:
        LOGGER.debug(
            "Lock adaptativo adquirido en predictive_service (span=%s, símbolos=%s)",
            span,
            symbols,
        )
        if active_cache is not None:
            cached = active_cache.get(cache_key)
        else:
            cached = None
        if isinstance(cached, pd.DataFrame):
            cached_frame = cached
            cache_hit = True
            telemetry["cache"] = "hit"
            last_updated = (
                _cache_last_updated(active_cache)
                if isinstance(active_cache, CacheService)
                else None
            )
            _CACHE_STATE.record_hit(
                last_updated=last_updated,
                ttl_hours=effective_ttl_hours,
            )
        LOGGER.debug(
            "Liberando lock adaptativo en predictive_service tras consulta de caché"
        )

    if cache_hit and isinstance(cached_frame, pd.DataFrame):
        LOGGER.debug(
            "Predicciones recuperadas desde caché (span=%s, símbolos=%s, sectores=%s)",
            span,
            symbols,
            sectors,
        )
        return cached_frame.copy()

    telemetry["cache"] = "miss"

    if frame.empty:
        empty = pd.DataFrame(
            columns=[
                "sector",
                "predicted_return",
                "sample_size",
                "avg_correlation",
                "confidence",
            ]
        )
        with adaptive_cache_lock:
            LOGGER.debug(
                "Lock adaptativo adquirido para persistir predicciones vacías (span=%s)",
                span,
            )
            if active_cache is not None:
                active_cache.set(cache_key, empty, ttl=ttl_seconds)
            last_updated = (
                _cache_last_updated(active_cache)
                if isinstance(active_cache, CacheService)
                else None
            )
            _CACHE_STATE.record_miss(
                last_updated=last_updated,
                ttl_hours=effective_ttl_hours,
            )
            LOGGER.debug(
                "Lock adaptativo liberado tras almacenar predicciones vacías"
            )
        telemetry["result_rows"] = 0
        return empty

    LOGGER.debug(
        "Ejecutando predictive engine %s (span=%s, símbolos=%s, sectores=%s)",
        ENGINE_VERSION,
        span,
        symbols,
        sectors,
    )

    service = backtesting_service or BacktestingService()

    with performance_timer("predictive_compute", extra=telemetry):
        predictions = build_sector_prediction_frame(
            frame,
            backtesting_service=service,
            run_backtest=lambda svc, symbol: run_backtest(svc, symbol, logger=LOGGER),
            extract_series=lambda backtest, column: extract_backtest_series(backtest, column),
            ema_predictor=lambda series, ema_span: compute_ema_prediction(series, span=ema_span),
            average_correlation=average_correlation,
            span=span,
        )

    telemetry["result_rows"] = int(len(predictions))

    stored_frame: pd.DataFrame = predictions
    reused = False

    with adaptive_cache_lock:
        LOGGER.debug(
            "Lock adaptativo adquirido para actualizar caché predictivo (span=%s)",
            span,
        )
        existing = active_cache.get(cache_key) if active_cache is not None else None
        if isinstance(existing, pd.DataFrame):
            stored_frame = existing
            reused = True
            telemetry["cache"] = "hit"
            last_updated = (
                _cache_last_updated(active_cache)
                if isinstance(active_cache, CacheService)
                else None
            )
            _CACHE_STATE.record_hit(
                last_updated=last_updated,
                ttl_hours=effective_ttl_hours,
            )
        else:
            if active_cache is not None:
                active_cache.set(cache_key, predictions, ttl=ttl_seconds)
            last_updated = (
                _cache_last_updated(active_cache)
                if isinstance(active_cache, CacheService)
                else None
            )
            _CACHE_STATE.record_miss(
                last_updated=last_updated,
                ttl_hours=effective_ttl_hours,
            )
        LOGGER.debug(
            "Lock adaptativo liberado tras actualizar caché predictivo"
        )

    if reused:
        LOGGER.debug(
            "Predicciones ya presentes en caché reutilizadas (span=%s, símbolos=%s)",
            span,
            symbols,
        )
    else:
        LOGGER.debug(
            "Predicciones almacenadas en caché (span=%s, símbolos=%s, sectores=%s)",
            span,
            symbols,
            sectors,
        )

    return stored_frame.copy()


def update_cache_metrics(cache: CacheService | None = None) -> PredictiveSnapshot:
    """Refresh predictive cache metrics without recomputing predictions."""

    active_cache: CacheService | None = cache if cache is not None else _CACHE
    with adaptive_cache_lock:
        LOGGER.debug(
            "Lock adaptativo adquirido para refrescar métricas de caché predictivo"
        )
        hits = _CACHE_STATE.hits
        misses = _CACHE_STATE.misses
        last_updated = _CACHE_STATE.last_updated or "-"
        ttl_hours = _CACHE_STATE.ttl_hours
        remaining_ttl: float | None = None

        if isinstance(active_cache, CacheService):
            try:
                hits = int(getattr(active_cache, "hits", hits))
            except Exception:  # pragma: no cover - defensive
                hits = _CACHE_STATE.hits
            try:
                misses = int(getattr(active_cache, "misses", misses))
            except Exception:  # pragma: no cover - defensive
                misses = _CACHE_STATE.misses
            cache_last_updated = _cache_last_updated(active_cache)
            if cache_last_updated:
                last_updated = cache_last_updated
            try:
                effective_ttl = active_cache.get_effective_ttl()
            except Exception:  # pragma: no cover - defensive
                effective_ttl = None
            if effective_ttl is not None:
                ttl_hours = float(effective_ttl) / 3600.0
            try:
                remaining_ttl = active_cache.remaining_ttl()
            except Exception:  # pragma: no cover - defensive safeguard
                remaining_ttl = None

        _CACHE_STATE.hits = hits
        _CACHE_STATE.misses = misses
        _CACHE_STATE.last_updated = last_updated or "-"
        _CACHE_STATE.ttl_hours = ttl_hours

        snapshot = PredictiveSnapshot(
            hits=hits,
            misses=misses,
            last_updated=_CACHE_STATE.last_updated,
            ttl_hours=_CACHE_STATE.ttl_hours,
            remaining_ttl=remaining_ttl,
        )

        LOGGER.debug(
            "Lock adaptativo liberado tras refrescar métricas de caché predictivo"
        )
    return snapshot


def get_cache_stats() -> PredictiveSnapshot:
    """Expose current cache counters for predictive workloads."""

    with adaptive_cache_lock:
        last_updated = _CACHE_STATE.last_updated
        if not last_updated or last_updated == "-":
            cache_last = _cache_last_updated(_CACHE)
            if cache_last:
                last_updated = cache_last
            else:
                last_updated = "-"

        ttl_hours = _CACHE_STATE.ttl_hours
        if ttl_hours is None:
            try:
                effective_ttl = _CACHE.get_effective_ttl()
            except Exception:  # pragma: no cover - defensive
                effective_ttl = None
            if effective_ttl is not None:
                ttl_hours = float(effective_ttl) / 3600.0

        remaining_ttl: float | None = None
        try:
            remaining_ttl = _CACHE.remaining_ttl()
        except Exception:  # pragma: no cover - defensive safeguard
            remaining_ttl = None

        snapshot = PredictiveSnapshot(
            hits=_CACHE_STATE.hits,
            misses=_CACHE_STATE.misses,
            last_updated=last_updated,
            ttl_hours=ttl_hours,
            remaining_ttl=remaining_ttl,
        )
    return snapshot


def reset_cache() -> None:
    """Utility mainly intended for tests to clear predictive caches."""

    global _CACHE_STATE
    with adaptive_cache_lock:
        LOGGER.debug("Lock adaptativo adquirido para reiniciar el caché predictivo")
        _CACHE.clear()
        _CACHE.set_ttl_override(PREDICTIVE_TTL_HOURS * 3600.0)
        _CACHE_STATE = PredictiveCacheState()
        LOGGER.debug("Lock adaptativo liberado tras reiniciar el caché predictivo")


__all__ = [
    "PredictiveSnapshot",
    "predict_sector_performance",
    "update_cache_metrics",
    "get_cache_stats",
    "reset_cache",
]
