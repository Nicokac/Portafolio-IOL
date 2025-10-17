from __future__ import annotations

import csv
import hashlib
import json
import logging
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Mapping, MutableMapping, Sequence

import numpy as np

import pandas as pd

from application.portfolio_service import PortfolioTotals, calculate_totals
from services import health
from services import snapshots as snapshot_service
from services.cache.market_data_cache import get_market_data_cache
from application.risk_service import (
    annualized_volatility,
    beta,
    compute_returns,
    max_drawdown,
)
from shared.telemetry import log_default_telemetry

logger = logging.getLogger(__name__)

_INCREMENTAL_METRICS_PATH = Path("performance_metrics_12.csv")
_INCREMENTAL_METRICS_FIELDS = (
    "dataset_hash",
    "filters_changed",
    "reused_blocks",
    "recomputed_blocks",
    "total_duration_s",
    "memoization_hit_ratio",
)


@dataclass(frozen=True)
class PortfolioCacheMetricsSnapshot:
    """Resumen inmutable del estado del memoizador del portafolio."""

    portfolio_view_render_s: float
    portfolio_cache_hit_ratio: float
    portfolio_cache_miss_count: int
    hits: int
    misses: int
    render_invocations: int
    fingerprint_invalidations: Dict[str, int]
    cache_miss_reasons: Dict[str, int]
    recent_misses: tuple[Dict[str, Any], ...]
    recent_invalidations: tuple[Dict[str, Any], ...]

    def total_invalidations(self) -> int:
        return sum(self.fingerprint_invalidations.values())

    def unnecessary_misses(self) -> int:
        return int(self.cache_miss_reasons.get("unchanged_fingerprint", 0) or 0)


class _PortfolioCacheTelemetry:
    """Recolecta métricas de uso del memoizador del portafolio."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recent_misses: Deque[Dict[str, Any]] = deque(maxlen=25)
        self._recent_invalidations: Deque[Dict[str, Any]] = deque(maxlen=25)
        self._reset_locked()

    def _reset_locked(self) -> None:
        self._render_total = 0.0
        self._render_invocations = 0
        self._hits = 0
        self._misses = 0
        self._invalidations: Counter[str] = Counter()
        self._miss_reasons: Counter[str] = Counter()
        self._recent_misses.clear()
        self._recent_invalidations.clear()

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()

    @staticmethod
    def _categorize_reason(dataset_changed: bool, filters_changed: bool) -> str:
        if dataset_changed and filters_changed:
            return "dataset_and_filters"
        if dataset_changed:
            return "dataset_changed"
        if filters_changed:
            return "filters_changed"
        return "unchanged_fingerprint"

    def record_hit(
        self,
        *,
        elapsed_s: float,
        dataset_changed: bool,
        filters_changed: bool,
    ) -> None:
        del dataset_changed, filters_changed
        with self._lock:
            self._render_total += max(float(elapsed_s), 0.0)
            self._render_invocations += 1
            self._hits += 1

    def record_miss(
        self,
        *,
        elapsed_s: float,
        dataset_changed: bool,
        filters_changed: bool,
        apply_elapsed: float,
        totals_elapsed: float,
    ) -> None:
        reason = self._categorize_reason(dataset_changed, filters_changed)
        event = {
            "ts": time.time(),
            "reason": reason,
            "dataset_changed": bool(dataset_changed),
            "filters_changed": bool(filters_changed),
            "apply_elapsed": max(float(apply_elapsed), 0.0),
            "totals_elapsed": max(float(totals_elapsed), 0.0),
            "render_elapsed": max(float(elapsed_s), 0.0),
        }
        with self._lock:
            self._render_total += max(float(elapsed_s), 0.0)
            self._render_invocations += 1
            self._misses += 1
            self._miss_reasons[reason] += 1
            self._recent_misses.append(event)

    def record_invalidation(self, reason: str, *, detail: str | None = None) -> None:
        reason_key = str(reason or "unknown").strip() or "unknown"
        detail_text = None
        if detail is not None:
            detail_text = str(detail).strip()
            if len(detail_text) > 120:
                detail_text = detail_text[:117] + "..."
        event = {"ts": time.time(), "reason": reason_key}
        if detail_text:
            event["detail"] = detail_text
        with self._lock:
            self._invalidations[reason_key] += 1
            self._recent_invalidations.append(event)

    def snapshot(self) -> PortfolioCacheMetricsSnapshot:
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = (self._hits / total) * 100.0 if total else 0.0
            return PortfolioCacheMetricsSnapshot(
                portfolio_view_render_s=self._render_total,
                portfolio_cache_hit_ratio=hit_ratio,
                portfolio_cache_miss_count=self._misses,
                hits=self._hits,
                misses=self._misses,
                render_invocations=self._render_invocations,
                fingerprint_invalidations=dict(self._invalidations),
                cache_miss_reasons=dict(self._miss_reasons),
                recent_misses=tuple(dict(item) for item in self._recent_misses),
                recent_invalidations=tuple(
                    dict(item) for item in self._recent_invalidations
                ),
            )


_PORTFOLIO_CACHE_TELEMETRY = _PortfolioCacheTelemetry()


def reset_portfolio_cache_metrics() -> None:
    """Reinicia las métricas recopiladas del memoizador del portafolio."""

    _PORTFOLIO_CACHE_TELEMETRY.reset()


def get_portfolio_cache_metrics_snapshot() -> PortfolioCacheMetricsSnapshot:
    """Devuelve un snapshot del estado actual del memoizador."""

    return _PORTFOLIO_CACHE_TELEMETRY.snapshot()


def _normalize_fingerprint_value(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, (pd.Series, pd.Index)):
        return _normalize_fingerprint_value(value.tolist())
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        return {str(k): _normalize_fingerprint_value(v) for k, v in items}
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_fingerprint_value(v) for v in value]
        try:
            return sorted(normalized)
        except TypeError:
            return normalized
    if isinstance(value, (list, tuple)):
        normalized = [_normalize_fingerprint_value(v) for v in value]
        try:
            return sorted(normalized)
        except TypeError:
            return normalized
    return value


def _fingerprint_from_payload(payload: Mapping[str, Any]) -> str:
    try:
        normalized = {
            str(key): _normalize_fingerprint_value(value)
            for key, value in payload.items()
        }
        serialized = json.dumps(normalized, sort_keys=True, default=_coerce_json)
    except Exception:
        serialized = json.dumps(
            {str(key): str(value) for key, value in payload.items()},
            sort_keys=True,
        )
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _extract_controls_attributes(controls: Any) -> dict[str, Any]:
    if controls is None:
        return {}
    if hasattr(controls, "__dict__"):
        return {
            key: value
            for key, value in vars(controls).items()
            if not key.startswith("_")
        }
    result: dict[str, Any] = {}
    for attr in dir(controls):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(controls, attr)
        except Exception:
            continue
        if callable(value):
            continue
        result[attr] = value
    return result


def _build_incremental_fingerprints(
    dataset_key: str, controls: Any, filters_key: str
) -> dict[str, str]:
    attributes = _extract_controls_attributes(controls)
    fingerprints: dict[str, str] = {
        "dataset": str(dataset_key or "empty"),
        "filters.base": str(filters_key or "none"),
    }

    time_payload = {
        key: attributes[key]
        for key in attributes
        if any(token in key.lower() for token in ("date", "range", "window", "period"))
    }
    fx_payload = {
        key: attributes[key]
        for key in attributes
        if any(token in key.lower() for token in ("fx", "currency", "exchange"))
    }
    misc_payload = {
        key: attributes[key]
        for key in attributes
        if key not in time_payload and key not in fx_payload
        and key not in {"hide_cash", "selected_syms", "selected_types", "symbol_query"}
    }

    fingerprints["filters.time"] = (
        _fingerprint_from_payload(time_payload) if time_payload else "none"
    )
    fingerprints["filters.fx"] = (
        _fingerprint_from_payload(fx_payload) if fx_payload else "none"
    )
    fingerprints["filters.misc"] = (
        _fingerprint_from_payload(misc_payload) if misc_payload else "none"
    )

    return fingerprints


def _compute_returns_block(df_view: pd.DataFrame) -> pd.DataFrame:
    if df_view is None or df_view.empty:
        return pd.DataFrame(columns=["simbolo", "return_pct"])

    df = df_view.copy(deep=False)
    if "pl_pct" in df.columns:
        returns = pd.DataFrame(
            {
                "simbolo": df.get("simbolo", pd.Series(dtype=str)),
                "return_pct": pd.to_numeric(df["pl_pct"], errors="coerce"),
            }
        )
        return returns.reset_index(drop=True)

    if {"pl", "costo"}.issubset(df.columns):
        costo = pd.to_numeric(df["costo"], errors="coerce")
        pl = pd.to_numeric(df["pl"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.isfinite(costo) & (np.abs(costo) > 1e-9), (pl / costo) * 100.0, np.nan)
        returns = pd.DataFrame({"return_pct": ratio})
        if "simbolo" in df.columns:
            returns["simbolo"] = df["simbolo"].values
        columns = [col for col in ("simbolo", "return_pct") if col in returns.columns]
        return returns.loc[:, columns].reset_index(drop=True)

    return pd.DataFrame(columns=["simbolo", "return_pct"])


def _should_reuse_block(
    *,
    dataset_changed: bool,
    previous_snapshot: PortfolioViewSnapshot | None,
    previous_fingerprints: Mapping[str, str],
    current_fingerprints: Mapping[str, str],
    dependencies: Sequence[str],
) -> bool:
    if dataset_changed or previous_snapshot is None:
        return False
    for dep in dependencies:
        if current_fingerprints.get(dep) != previous_fingerprints.get(dep):
            return False
    return True


def compute_incremental_view(
    *,
    dataset_changed: bool,
    fingerprints: Mapping[str, str],
    previous_snapshot: PortfolioViewSnapshot | None,
    previous_blocks: Mapping[str, Any] | None,
    load_positions: Callable[[], tuple[pd.DataFrame, float]],
    compute_totals_fn: Callable[[pd.DataFrame], PortfolioTotals],
    compute_contributions_fn: Callable[[pd.DataFrame], PortfolioContributionMetrics],
    update_history_fn: Callable[[PortfolioTotals], pd.DataFrame],
    build_returns_fn: Callable[[pd.DataFrame], pd.DataFrame] = _compute_returns_block,
) -> IncrementalComputationResult:
    start = time.perf_counter()
    prev_blocks: MutableMapping[str, Any] = dict(previous_blocks or {})
    prev_fingerprints: Mapping[str, str] = prev_blocks.get("fingerprints", {})
    reused_blocks: set[str] = set()
    recomputed_blocks: set[str] = set()

    positions_df: pd.DataFrame | None = None
    apply_elapsed = 0.0
    if _should_reuse_block(
        dataset_changed=dataset_changed,
        previous_snapshot=previous_snapshot,
        previous_fingerprints=prev_fingerprints,
        current_fingerprints=fingerprints,
        dependencies=("dataset", "filters.base", "filters.misc"),
    ):
        cached_positions = prev_blocks.get("positions_df")
        if isinstance(cached_positions, pd.DataFrame):
            positions_df = cached_positions
            reused_blocks.add("positions_df")

    if positions_df is None:
        positions_df, apply_elapsed = load_positions()
        if positions_df is None:
            positions_df = pd.DataFrame()
        recomputed_blocks.add("positions_df")

    if not isinstance(positions_df, pd.DataFrame):
        positions_df = pd.DataFrame(positions_df)

    returns_df: pd.DataFrame | None = None
    if _should_reuse_block(
        dataset_changed=dataset_changed,
        previous_snapshot=previous_snapshot,
        previous_fingerprints=prev_fingerprints,
        current_fingerprints=fingerprints,
        dependencies=("dataset", "filters.time", "filters.fx"),
    ) and "positions_df" in reused_blocks:
        cached_returns = prev_blocks.get("returns_df")
        if isinstance(cached_returns, pd.DataFrame):
            returns_df = cached_returns
            reused_blocks.add("returns_df")

    if returns_df is None:
        returns_df = build_returns_fn(positions_df)
        if returns_df is None:
            returns_df = pd.DataFrame()
        recomputed_blocks.add("returns_df")

    totals: PortfolioTotals | None = None
    totals_elapsed = 0.0
    if _should_reuse_block(
        dataset_changed=dataset_changed,
        previous_snapshot=previous_snapshot,
        previous_fingerprints=prev_fingerprints,
        current_fingerprints=fingerprints,
        dependencies=("dataset", "filters.base", "filters.time", "filters.fx", "filters.misc"),
    ) and "positions_df" in reused_blocks:
        totals = previous_snapshot.totals
        reused_blocks.add("totals")

    if totals is None:
        totals_start = time.perf_counter()
        totals = compute_totals_fn(positions_df)
        totals_elapsed = time.perf_counter() - totals_start
        recomputed_blocks.add("totals")

    contribution_metrics: PortfolioContributionMetrics | None = None
    if _should_reuse_block(
        dataset_changed=dataset_changed,
        previous_snapshot=previous_snapshot,
        previous_fingerprints=prev_fingerprints,
        current_fingerprints=fingerprints,
        dependencies=("dataset", "filters.base", "filters.misc"),
    ) and "positions_df" in reused_blocks:
        contribution_metrics = previous_snapshot.contribution_metrics
        reused_blocks.add("contribution_metrics")

    if contribution_metrics is None:
        contribution_metrics = compute_contributions_fn(positions_df)
        recomputed_blocks.add("contribution_metrics")

    history_df = update_history_fn(totals)
    duration = time.perf_counter() - start

    return IncrementalComputationResult(
        df_view=positions_df,
        totals=totals,
        contribution_metrics=contribution_metrics,
        historical_total=history_df,
        returns_df=returns_df,
        apply_elapsed=apply_elapsed,
        totals_elapsed=totals_elapsed,
        reused_blocks=tuple(sorted(reused_blocks)),
        recomputed_blocks=tuple(sorted(recomputed_blocks)),
        duration=duration,
    )


def _append_incremental_metric(
    *,
    dataset_hash: str,
    filters_changed: bool,
    reused_blocks: Sequence[str],
    recomputed_blocks: Sequence[str],
    total_duration: float,
    memoization_hit_ratio: float,
) -> None:
    payload = {
        "dataset_hash": str(dataset_hash),
        "filters_changed": "true" if filters_changed else "false",
        "reused_blocks": ";".join(sorted(reused_blocks)),
        "recomputed_blocks": ";".join(sorted(recomputed_blocks)),
        "total_duration_s": f"{max(float(total_duration), 0.0):.6f}",
        "memoization_hit_ratio": f"{max(min(memoization_hit_ratio, 1.0), 0.0):.3f}",
    }
    try:
        _INCREMENTAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_exists = _INCREMENTAL_METRICS_PATH.exists()
        with _INCREMENTAL_METRICS_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_INCREMENTAL_METRICS_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(payload)
    except Exception:  # pragma: no cover - best effort logging
        logger.debug(
            "No se pudo actualizar %s con métricas incrementales",
            _INCREMENTAL_METRICS_PATH,
            exc_info=True,
        )

_SYMBOL_RISK_CACHE: dict[tuple[str, str, str], tuple[dict[str, Any], str, float]] = {}
_SYMBOL_RISK_CACHE_MAX = 512


def _series_signature(series: pd.Series | None) -> str:
    if series is None or series.empty:
        return "empty"
    try:
        hashed = pd.util.hash_pandas_object(series, index=True, categorize=False)
        return hashlib.sha1(hashed.values.tobytes()).hexdigest()
    except Exception:
        try:
            values = np.asarray(series.to_numpy(copy=False))
            return hashlib.sha1(values.tobytes()).hexdigest()
        except Exception:
            tail_value = series.iloc[-1] if len(series) else None
            return f"len:{len(series)}|tail:{tail_value}"


def _prune_symbol_risk_cache() -> None:
    if len(_SYMBOL_RISK_CACHE) <= _SYMBOL_RISK_CACHE_MAX:
        return
    surplus = len(_SYMBOL_RISK_CACHE) - _SYMBOL_RISK_CACHE_MAX
    if surplus <= 0:
        return
    ordered_keys = sorted(
        _SYMBOL_RISK_CACHE.items(), key=lambda item: item[1][2]
    )
    for key, _ in ordered_keys[:surplus]:
        _SYMBOL_RISK_CACHE.pop(key, None)

def compute_symbol_risk_metrics(
    tasvc,
    symbols: list[str],
    *,
    benchmark: str,
    period: str,
) -> pd.DataFrame:
    """Return risk metrics (volatility, drawdown, beta) for each symbol.

    Parameters
    ----------
    tasvc:
        Service exposing ``portfolio_history``.
    symbols:
        Portfolio symbols to evaluate.
    benchmark:
        Benchmark symbol used to compare beta and relative metrics.
    period:
        Historical period used to compute metrics (e.g. ``"6mo"``).
    """

    if tasvc is None or not symbols:
        return pd.DataFrame()

    request_symbols = list({*symbols, benchmark})
    if not request_symbols:
        return pd.DataFrame()

    cache = get_market_data_cache()
    try:
        prices = cache.get_history(
            request_symbols,
            loader=lambda symbols=request_symbols: tasvc.portfolio_history(
                simbolos=list(symbols), period=period
            ),
            period=period,
            benchmark=benchmark,
        )
    except Exception:
        logger.exception("Error fetching portfolio history for risk metrics")
        return pd.DataFrame()

    if prices is None or prices.empty:
        return pd.DataFrame()

    returns = compute_returns(prices)
    if returns.empty:
        return pd.DataFrame()

    metrics: list[dict[str, Any]] = []

    bench_returns = returns.get(benchmark)
    if bench_returns is None:
        bench_returns = pd.Series(dtype=float)

    benchmark_key = str(benchmark or "").strip().upper()
    period_key = str(period or "").strip()
    bench_signature = _series_signature(prices.get(benchmark))

    for sym in prices.columns:
        sym_returns = returns.get(sym)
        if sym_returns is None or sym_returns.empty:
            continue

        norm_symbol = str(sym or "").strip().upper()
        cache_key = (norm_symbol, benchmark_key, period_key)
        sym_signature = _series_signature(prices.get(sym))
        combined_signature = f"{sym_signature}|{bench_signature}"
        cached_entry = _SYMBOL_RISK_CACHE.get(cache_key)
        if cached_entry and cached_entry[1] == combined_signature:
            payload = dict(cached_entry[0])
            metrics.append(payload)
            _SYMBOL_RISK_CACHE[cache_key] = (
                dict(payload),
                combined_signature,
                time.time(),
            )
            continue

        vol = annualized_volatility(sym_returns)
        dd = max_drawdown(sym_returns)

        is_benchmark = sym == benchmark
        if is_benchmark:
            sym_beta = 1.0 if len(sym_returns) else float("nan")
        elif bench_returns.empty:
            sym_beta = float("nan")
        else:
            aligned_sym, aligned_bench = sym_returns.align(
                bench_returns, join="inner"
            )
            sym_beta = beta(aligned_sym, aligned_bench)

        record = {
            "simbolo": sym,
            "volatilidad": vol,
            "drawdown": dd,
            "beta": sym_beta,
            "es_benchmark": is_benchmark,
        }
        metrics.append(record)
        _SYMBOL_RISK_CACHE[cache_key] = (
            dict(record),
            combined_signature,
            time.time(),
        )

    _prune_symbol_risk_cache()

    if not metrics:
        return pd.DataFrame()

    metrics_df = pd.DataFrame(metrics)
    for col in ("volatilidad", "drawdown", "beta"):
        if col in metrics_df.columns:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce").astype(
                "float32"
            )
    return metrics_df


@dataclass(frozen=True)
class PortfolioContributionMetrics:
    """Aggregate contribution metrics for charts and analytics."""

    by_symbol: pd.DataFrame
    by_type: pd.DataFrame

    @classmethod
    def empty(cls) -> "PortfolioContributionMetrics":
        cols = [
            "tipo",
            "simbolo",
            "valor_actual",
            "costo",
            "pl",
            "pl_d",
            "valor_actual_pct",
            "pl_pct",
        ]
        by_symbol = pd.DataFrame(columns=cols)
        by_type = pd.DataFrame(
            columns=[
                "tipo",
                "valor_actual",
                "costo",
                "pl",
                "pl_d",
                "valor_actual_pct",
                "pl_pct",
            ]
        )
        return cls(by_symbol=by_symbol, by_type=by_type)


@dataclass(frozen=True)
class PortfolioViewSnapshot:
    """Resultado cacheado del portafolio."""

    df_view: pd.DataFrame
    totals: PortfolioTotals
    apply_elapsed: float
    totals_elapsed: float
    generated_at: float
    historical_total: pd.DataFrame
    contribution_metrics: PortfolioContributionMetrics
    storage_id: str | None = None


@dataclass(frozen=True)
class IncrementalComputationResult:
    """Resultado de `compute_incremental_view` con telemetría asociada."""

    df_view: pd.DataFrame
    totals: PortfolioTotals
    contribution_metrics: PortfolioContributionMetrics
    historical_total: pd.DataFrame
    returns_df: pd.DataFrame
    apply_elapsed: float
    totals_elapsed: float
    reused_blocks: tuple[str, ...]
    recomputed_blocks: tuple[str, ...]
    duration: float


class PortfolioViewModelService:
    """Wrapper cacheado alrededor de ``apply_filters``.

    Memoriza el último resultado junto con los totales derivados para evitar
    recomputar mientras no cambien ni las posiciones ni los filtros
    relevantes.
    """

    def __init__(self, *, snapshot_backend: Any | None = None) -> None:
        self._snapshot: PortfolioViewSnapshot | None = None
        self._dataset_key: str | None = None
        self._filters_key: str | None = None
        self._history_records: list[dict[str, float]] = []
        self._incremental_cache: dict[str, Any] = {}
        self._last_incremental_stats: dict[str, Any] | None = None
        self._snapshot_kind = "portfolio"
        self.configure_snapshot_backend(snapshot_backend)

    def configure_snapshot_backend(self, snapshot_backend: Any | None) -> None:
        """Configure the storage backend used to persist portfolio snapshots."""

        if snapshot_backend is None:
            self._snapshot_storage = snapshot_service
        else:
            self._snapshot_storage = snapshot_backend

    def _snapshot_backend_details(self) -> Dict[str, Any]:
        backend = getattr(self, "_snapshot_storage", None)
        details: Dict[str, Any] = {}
        if backend is None:
            return details

        backend_name: str | None = None
        getter = getattr(backend, "current_backend_name", None)
        if callable(getter):
            try:
                backend_name = getter()
            except Exception:
                logger.debug("No se pudo determinar el backend activo de snapshots", exc_info=True)
        if isinstance(backend_name, str):
            backend_name = backend_name.strip() or None
        elif backend_name is not None:
            backend_name = str(backend_name)

        if not backend_name:
            raw_name = getattr(backend, "backend_name", None)
            if isinstance(raw_name, str):
                backend_name = raw_name.strip() or None
            elif raw_name is not None:
                backend_name = str(raw_name)

        if not backend_name:
            module_name = getattr(backend, "__name__", None)
            if isinstance(module_name, str) and module_name.strip():
                backend_name = module_name.strip()

        if not backend_name:
            backend_name = backend.__class__.__name__

        if backend_name:
            details["name"] = str(backend_name)

        for attr in ("path", "storage_path", "location"):
            value = getattr(backend, attr, None)
            if value:
                details[attr] = str(value)

        return details

    def _record_snapshot_event(
        self,
        *,
        action: str,
        status: str,
        storage_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        try:
            health.record_snapshot_event(
                kind=self._snapshot_kind,
                action=action,
                status=status,
                storage_id=storage_id,
                detail=detail,
                backend=self._snapshot_backend_details(),
            )
        except Exception:
            logger.debug("No se pudo registrar el evento de snapshot", exc_info=True)

    def _update_history(self, totals: PortfolioTotals) -> pd.DataFrame:
        entry = _history_row(time.time(), totals)
        self._history_records = _append_history(self._history_records, entry, maxlen=500)
        return _normalize_history_df(self._history_records)

    @staticmethod
    def _hash_dataset(df: pd.DataFrame | None) -> str:
        if df is None or df.empty:
            return "empty"
        try:
            hashed = pd.util.hash_pandas_object(df, index=True, categorize=True)
            return hashlib.sha1(hashed.values.tobytes()).hexdigest()
        except TypeError:
            payload = json.dumps(
                df.to_dict(orient="list"), sort_keys=True, default=_coerce_json
            ).encode("utf-8")
            return hashlib.sha1(payload).hexdigest()

    @staticmethod
    def _filters_key_from(controls: Any) -> str:
        payload = {
            "hide_cash": getattr(controls, "hide_cash", None),
            "selected_syms": sorted(map(str, getattr(controls, "selected_syms", []))),
            "selected_types": sorted(
                map(str, getattr(controls, "selected_types", []))
            ),
            "symbol_query": (getattr(controls, "symbol_query", "") or "").strip(),
        }
        return json.dumps(payload, sort_keys=True)

    def _persist_snapshot(
        self,
        *,
        df_view: pd.DataFrame,
        totals: PortfolioTotals,
        controls: Any,
        dataset_key: str,
        filters_key: str,
        generated_at: float,
        contribution_metrics: PortfolioContributionMetrics,
        historical_total: pd.DataFrame,
    ) -> tuple[str | None, pd.DataFrame | None]:
        backend = getattr(self._snapshot_storage, "save_snapshot", None)
        list_fn = getattr(self._snapshot_storage, "list_snapshots", None)
        if not callable(backend):
            return None, None

        payload = _serialize_snapshot_payload(
            df_view=df_view,
            totals=totals,
            generated_at=generated_at,
            history=historical_total,
            contribution_metrics=contribution_metrics,
        )
        metadata = {
            "dataset_key": dataset_key,
            "filters_key": filters_key,
            "controls": _safe_asdict(controls),
        }

        try:
            saved = backend(self._snapshot_kind, payload, metadata)
        except Exception as exc:
            logger.exception("No se pudo persistir el snapshot del portafolio")
            self._record_snapshot_event(
                action="save",
                status="error",
                detail=str(exc),
            )
            return None, None

        storage_id = saved.get("id") if isinstance(saved, Mapping) else None
        self._record_snapshot_event(
            action="save",
            status="saved",
            storage_id=str(storage_id) if storage_id else None,
        )
        persisted_history: pd.DataFrame | None = None
        if callable(list_fn):
            try:
                records = list_fn(self._snapshot_kind, limit=500, order="asc")
                persisted_history = _history_df_from_snapshot_records(records)
                if isinstance(persisted_history, pd.DataFrame) and persisted_history.empty:
                    persisted_history = None
            except Exception:
                logger.exception("No se pudo construir la historia persistida del portafolio")

        return storage_id, persisted_history

    def invalidate_positions(self, dataset_key: str | None = None) -> None:
        """Invalida el snapshot cuando cambia el dataset base."""

        self._snapshot = None
        self._dataset_key = dataset_key
        self._filters_key = None
        self._history_records = []
        self._incremental_cache = {}
        self._last_incremental_stats = None
        _PORTFOLIO_CACHE_TELEMETRY.record_invalidation(
            "dataset_changed", detail=dataset_key
        )
        logger.info(
            "portfolio_view cache invalidated (positions) dataset=%s", dataset_key
        )

    def invalidate_filters(self, filters_key: str | None = None) -> None:
        """Invalida el snapshot cuando cambian los filtros relevantes."""

        self._snapshot = None
        self._filters_key = filters_key
        self._incremental_cache = {}
        self._last_incremental_stats = None
        _PORTFOLIO_CACHE_TELEMETRY.record_invalidation(
            "filters_changed", detail=filters_key
        )
        logger.info(
            "portfolio_view cache invalidated (filters) filters=%s", filters_key
        )

    def get_portfolio_view(self, df_pos, controls, cli, psvc) -> PortfolioViewSnapshot:
        """Devuelve el resultado de ``apply_filters`` con cacheo básico."""

        dataset_key = self._hash_dataset(df_pos)
        filters_key = self._filters_key_from(controls)

        dataset_changed = dataset_key != self._dataset_key
        filters_changed = filters_key != self._filters_key

        render_start = time.perf_counter()

        if dataset_changed:
            self.invalidate_positions(dataset_key)
        previous_snapshot = self._snapshot

        cached_blocks: Mapping[str, Any] = {}
        cached_fingerprints: Mapping[str, str] = {}
        if not dataset_changed:
            cached_blocks = dict(self._incremental_cache)
            cached_fp = cached_blocks.get("fingerprints")
            if isinstance(cached_fp, Mapping):
                cached_fingerprints = dict(cached_fp)

        fingerprints = _build_incremental_fingerprints(
            dataset_key, controls, filters_key
        )

        incremental_changed = any(
            fingerprints.get(key) != cached_fingerprints.get(key)
            for key in ("filters.time", "filters.fx", "filters.misc")
        )

        effective_filters_changed = filters_changed or incremental_changed

        if (
            previous_snapshot is not None
            and not dataset_changed
            and not effective_filters_changed
            and not incremental_changed
        ):
            snapshot = previous_snapshot
            self._record_snapshot_event(
                action="load",
                status="reused",
                storage_id=snapshot.storage_id,
            )
            logger.info(
                "portfolio_view cache hit dataset_changed=%s filters_changed=%s apply=%.4fs totals=%.4fs",
                dataset_changed,
                effective_filters_changed,
                snapshot.apply_elapsed,
                snapshot.totals_elapsed,
            )
            elapsed = time.perf_counter() - render_start
            _PORTFOLIO_CACHE_TELEMETRY.record_hit(
                elapsed_s=elapsed,
                dataset_changed=dataset_changed,
                filters_changed=effective_filters_changed,
            )
            try:
                log_default_telemetry(
                    phase="portfolio_view.apply",
                    elapsed_s=snapshot.apply_elapsed,
                    dataset_hash=dataset_key,
                    memo_hit_ratio=1.0,
                )
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug(
                    "No se pudo registrar telemetría para portfolio_view.apply (hit)",
                    exc_info=True,
                )
            return snapshot

        positions_state: dict[str, Any] = {}

        def _load_positions() -> tuple[pd.DataFrame, float]:
            if "df" not in positions_state:
                start = time.perf_counter()
                df = _apply_filters(df_pos, controls, cli, psvc)
                positions_state["df"] = df
                positions_state["elapsed"] = time.perf_counter() - start
            return (
                positions_state.get("df", pd.DataFrame()),
                float(positions_state.get("elapsed", 0.0)),
            )

        incremental = compute_incremental_view(
            dataset_changed=dataset_changed,
            fingerprints=fingerprints,
            previous_snapshot=previous_snapshot,
            previous_blocks=cached_blocks,
            load_positions=_load_positions,
            compute_totals_fn=calculate_totals,
            compute_contributions_fn=_compute_contribution_metrics,
            update_history_fn=self._update_history,
        )

        df_view = incremental.df_view
        totals = incremental.totals
        apply_elapsed = incremental.apply_elapsed
        totals_elapsed = incremental.totals_elapsed
        contribution_metrics = incremental.contribution_metrics
        history_df = incremental.historical_total

        generated_ts = time.time()
        storage_id, persisted_history = self._persist_snapshot(
            df_view=df_view,
            totals=totals,
            controls=controls,
            dataset_key=dataset_key,
            filters_key=filters_key,
            generated_at=generated_ts,
            contribution_metrics=contribution_metrics,
            historical_total=history_df,
        )

        if persisted_history is not None:
            history_df = persisted_history

        snapshot = PortfolioViewSnapshot(
            df_view=df_view,
            totals=totals,
            apply_elapsed=apply_elapsed,
            totals_elapsed=totals_elapsed,
            generated_at=generated_ts,
            historical_total=history_df,
            contribution_metrics=contribution_metrics,
            storage_id=storage_id,
        )

        self._snapshot = snapshot
        self._dataset_key = dataset_key
        self._filters_key = filters_key
        self._incremental_cache = {
            "positions_df": df_view,
            "returns_df": incremental.returns_df,
            "fingerprints": dict(fingerprints),
        }

        total_blocks = len(incremental.reused_blocks) + len(incremental.recomputed_blocks)
        hit_ratio = (len(incremental.reused_blocks) / total_blocks) if total_blocks else 0.0
        self._last_incremental_stats = {
            "reused_blocks": incremental.reused_blocks,
            "recomputed_blocks": incremental.recomputed_blocks,
            "memoization_hit_ratio": hit_ratio,
            "dataset_changed": dataset_changed,
            "filters_changed": effective_filters_changed,
            "compute_duration_s": incremental.duration,
            "apply_elapsed": apply_elapsed,
            "totals_elapsed": totals_elapsed,
        }

        logger.info(
            "portfolio_view incremental update dataset_changed=%s filters_changed=%s reused=%s recomputed=%s apply=%.4fs totals=%.4fs",
            dataset_changed,
            effective_filters_changed,
            ",".join(incremental.reused_blocks) or "none",
            ",".join(incremental.recomputed_blocks) or "none",
            apply_elapsed,
            totals_elapsed,
        )
        total_elapsed = time.perf_counter() - render_start
        _PORTFOLIO_CACHE_TELEMETRY.record_miss(
            elapsed_s=total_elapsed,
            dataset_changed=dataset_changed,
            filters_changed=effective_filters_changed,
            apply_elapsed=apply_elapsed,
            totals_elapsed=totals_elapsed,
        )
        _append_incremental_metric(
            dataset_hash=dataset_key,
            filters_changed=effective_filters_changed,
            reused_blocks=incremental.reused_blocks,
            recomputed_blocks=incremental.recomputed_blocks,
            total_duration=total_elapsed,
            memoization_hit_ratio=hit_ratio,
        )
        try:
            log_default_telemetry(
                phase="portfolio_view.apply",
                elapsed_s=apply_elapsed,
                dataset_hash=dataset_key,
                memo_hit_ratio=hit_ratio,
            )
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug(
                "No se pudo registrar telemetría para portfolio_view.apply",
                exc_info=True,
            )
        return snapshot


def _coerce_json(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    return str(value)


def _apply_filters(df_pos, controls, cli, psvc):
    from controllers.portfolio.filters import apply_filters as _apply

    return _apply(df_pos, controls, cli, psvc)


def _compute_contribution_metrics(df_view: pd.DataFrame) -> PortfolioContributionMetrics:
    if df_view is None or df_view.empty:
        return PortfolioContributionMetrics.empty()

    cols = {
        "valor_actual": "valor_actual",
        "costo": "costo",
        "pl": "pl",
        "pl_d": "pl_d",
    }
    df = df_view.copy()
    for src, dest in cols.items():
        if src in df.columns:
            df[dest] = pd.to_numeric(df[src], errors="coerce")
        else:
            df[dest] = np.nan

    required = ["tipo", "simbolo"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    numeric_cols = list(cols.values())

    by_symbol = (
        df.groupby(["tipo", "simbolo"], dropna=False)[numeric_cols]
        .sum(min_count=1)
        .reset_index()
    )

    total_val = by_symbol["valor_actual"].sum(min_count=1)
    total_pl = by_symbol["pl"].sum(min_count=1)

    if not np.isfinite(total_val) or np.isclose(total_val, 0.0):
        by_symbol["valor_actual_pct"] = np.nan
    else:
        by_symbol["valor_actual_pct"] = (by_symbol["valor_actual"] / total_val) * 100.0

    if not np.isfinite(total_pl) or np.isclose(total_pl, 0.0):
        by_symbol["pl_pct"] = np.nan
    else:
        by_symbol["pl_pct"] = (by_symbol["pl"] / total_pl) * 100.0

    by_type = (
        df.groupby("tipo", dropna=False)[numeric_cols]
        .sum(min_count=1)
        .reset_index()
    )

    if not np.isfinite(total_val) or np.isclose(total_val, 0.0):
        by_type["valor_actual_pct"] = np.nan
    else:
        by_type["valor_actual_pct"] = (by_type["valor_actual"] / total_val) * 100.0

    if not np.isfinite(total_pl) or np.isclose(total_pl, 0.0):
        by_type["pl_pct"] = np.nan
    else:
        by_type["pl_pct"] = (by_type["pl"] / total_pl) * 100.0

    return PortfolioContributionMetrics(by_symbol=by_symbol, by_type=by_type)


def _history_row(ts: float, totals: PortfolioTotals) -> dict[str, float]:
    return {
        "timestamp": ts,
        "total_value": float(totals.total_value),
        "total_cost": float(totals.total_cost),
        "total_pl": float(totals.total_pl),
    }


def _normalize_history_df(records: list[dict[str, float]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["timestamp", "total_value", "total_cost", "total_pl"])
    df = pd.DataFrame.from_records(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    return df


def _append_history(records: list[dict[str, float]], entry: dict[str, float], maxlen: int) -> list[dict[str, float]]:
    records.append(entry)
    if maxlen and maxlen > 0:
        records = records[-maxlen:]
    return records


def _serialize_snapshot_payload(
    *,
    df_view: pd.DataFrame,
    totals: PortfolioTotals,
    generated_at: float,
    history: pd.DataFrame,
    contribution_metrics: PortfolioContributionMetrics,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at": float(generated_at),
        "totals": {
            "total_value": float(totals.total_value),
            "total_cost": float(totals.total_cost),
            "total_pl": float(totals.total_pl),
            "total_pl_pct": float(totals.total_pl_pct),
            "total_cash": float(totals.total_cash),
        },
    }

    payload["df_view"] = _df_to_records(df_view)
    payload["history"] = _df_to_records(history)

    contrib = {}
    if isinstance(contribution_metrics, PortfolioContributionMetrics):
        contrib["by_symbol"] = _df_to_records(contribution_metrics.by_symbol)
        contrib["by_type"] = _df_to_records(contribution_metrics.by_type)
    payload["contribution_metrics"] = contrib
    return payload


def _df_to_records(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    try:
        return df.replace({np.nan: None}).to_dict(orient="records")
    except Exception:
        return []


def _safe_asdict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    try:
        return asdict(obj)
    except TypeError:
        result: dict[str, Any] = {}
        for key in dir(obj):
            if key.startswith("_"):
                continue
            try:
                value = getattr(obj, key)
            except Exception:
                continue
            if callable(value):
                continue
            result[key] = value
        return result


def _as_mapping(value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        result: dict[str, float] = {}
        for key, val in value.items():
            try:
                result[str(key)] = float(val)
            except (TypeError, ValueError):
                continue
        return result
    return {}


def _history_df_from_snapshot_records(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    history_rows: list[dict[str, float]] = []
    for row in records:
        payload = row.get("payload") if isinstance(row, Mapping) else {}
        totals = {}
        if isinstance(payload, Mapping):
            totals = payload.get("totals") or {}
        ts = row.get("created_at") if isinstance(row, Mapping) else None
        if not ts and isinstance(payload, Mapping):
            ts = payload.get("generated_at")
        entry = {
            "timestamp": float(ts or 0.0),
            "total_value": float(_get_numeric(totals, "total_value")),
            "total_cost": float(_get_numeric(totals, "total_cost")),
            "total_pl": float(_get_numeric(totals, "total_pl")),
        }
        history_rows.append(entry)
    return _normalize_history_df(history_rows)


def _get_numeric(payload: Mapping[str, Any], key: str) -> float:
    try:
        return float(payload.get(key, 0.0))
    except (TypeError, ValueError, AttributeError):
        return 0.0

