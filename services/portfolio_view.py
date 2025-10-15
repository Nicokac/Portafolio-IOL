from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Sequence

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

logger = logging.getLogger(__name__)

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
    comparison: "SnapshotComparison" | None = None


@dataclass(frozen=True)
class SnapshotComparison:
    """Resumen de la comparación contra un snapshot histórico."""

    reference_id: str
    reference_timestamp: float
    deltas: Mapping[str, float]
    reference_totals: Mapping[str, float]
    metadata: Mapping[str, Any]


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
    ) -> tuple[str | None, SnapshotComparison | None, pd.DataFrame | None]:
        backend = getattr(self._snapshot_storage, "save_snapshot", None)
        list_fn = getattr(self._snapshot_storage, "list_snapshots", None)
        compare_fn = getattr(self._snapshot_storage, "compare_snapshots", None)
        if not callable(backend):
            return None, None, None

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
            return None, None, None

        storage_id = saved.get("id") if isinstance(saved, Mapping) else None
        self._record_snapshot_event(
            action="save",
            status="saved",
            storage_id=str(storage_id) if storage_id else None,
        )
        comparison: SnapshotComparison | None = None

        if callable(compare_fn) and storage_id:
            try:
                history = list_fn(self._snapshot_kind, limit=2, order="desc") if callable(list_fn) else []
                previous = next(
                    (row for row in history if row.get("id") != storage_id),
                    None,
                )
                if previous and previous.get("id"):
                    cmp = compare_fn(storage_id, previous["id"])
                    if isinstance(cmp, Mapping):
                        comparison = SnapshotComparison(
                            reference_id=str(previous.get("id")),
                            reference_timestamp=float(previous.get("created_at") or 0.0),
                            deltas=_as_mapping(cmp.get("delta")),
                            reference_totals=_as_mapping(cmp.get("totals_b")),
                            metadata=_as_mapping(cmp.get("metadata_b")),
                        )
            except Exception:
                logger.exception("No se pudo calcular la comparación del snapshot")

        persisted_history: pd.DataFrame | None = None
        if callable(list_fn):
            try:
                records = list_fn(self._snapshot_kind, limit=500, order="asc")
                persisted_history = _history_df_from_snapshot_records(records)
                if isinstance(persisted_history, pd.DataFrame) and persisted_history.empty:
                    persisted_history = None
            except Exception:
                logger.exception("No se pudo construir la historia persistida del portafolio")

        return storage_id, comparison, persisted_history

    def invalidate_positions(self, dataset_key: str | None = None) -> None:
        """Invalida el snapshot cuando cambia el dataset base."""

        self._snapshot = None
        self._dataset_key = dataset_key
        self._filters_key = None
        self._history_records = []
        logger.info(
            "portfolio_view cache invalidated (positions) dataset=%s", dataset_key
        )

    def invalidate_filters(self, filters_key: str | None = None) -> None:
        """Invalida el snapshot cuando cambian los filtros relevantes."""

        self._snapshot = None
        self._filters_key = filters_key
        logger.info(
            "portfolio_view cache invalidated (filters) filters=%s", filters_key
        )

    def get_portfolio_view(self, df_pos, controls, cli, psvc) -> PortfolioViewSnapshot:
        """Devuelve el resultado de ``apply_filters`` con cacheo básico."""

        dataset_key = self._hash_dataset(df_pos)
        filters_key = self._filters_key_from(controls)

        dataset_changed = dataset_key != self._dataset_key
        filters_changed = filters_key != self._filters_key

        if dataset_changed:
            self.invalidate_positions(dataset_key)
        elif filters_changed:
            self.invalidate_filters(filters_key)

        if (
            self._snapshot is not None
            and dataset_key == self._dataset_key
            and filters_key == self._filters_key
        ):
            snapshot = self._snapshot
            self._record_snapshot_event(
                action="load",
                status="reused",
                storage_id=snapshot.storage_id,
            )
            logger.info(
                "portfolio_view cache hit dataset_changed=%s filters_changed=%s apply=%.4fs totals=%.4fs",
                dataset_changed,
                filters_changed,
                snapshot.apply_elapsed,
                snapshot.totals_elapsed,
            )
            return snapshot

        start = time.perf_counter()
        df_view = _apply_filters(df_pos, controls, cli, psvc)
        apply_elapsed = time.perf_counter() - start

        totals_start = time.perf_counter()
        totals = calculate_totals(df_view)
        totals_elapsed = time.perf_counter() - totals_start

        contribution_metrics = _compute_contribution_metrics(df_view)
        history_df = self._update_history(totals)

        generated_ts = time.time()
        storage_id, comparison, persisted_history = self._persist_snapshot(
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
            comparison=comparison,
        )

        self._snapshot = snapshot
        self._dataset_key = dataset_key
        self._filters_key = filters_key

        logger.info(
            "portfolio_view cache miss dataset_changed=%s filters_changed=%s apply=%.4fs totals=%.4fs",
            dataset_changed,
            filters_changed,
            apply_elapsed,
            totals_elapsed,
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

