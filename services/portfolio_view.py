from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

import pandas as pd

from application.portfolio_service import PortfolioTotals, calculate_totals
from application.risk_service import (
    annualized_volatility,
    beta,
    compute_returns,
)

logger = logging.getLogger(__name__)

def _max_drawdown_from_returns(returns: pd.Series) -> float:
    """Compute maximum drawdown given a return series."""

    if returns is None or returns.empty:
        return 0.0

    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    drawdowns = wealth / running_max - 1
    return float(drawdowns.min())


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

    try:
        prices = tasvc.portfolio_history(
            simbolos=list({*symbols, benchmark}), period=period
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

    for sym in prices.columns:
        sym_returns = returns.get(sym)
        if sym_returns is None or sym_returns.empty:
            continue

        vol = annualized_volatility(sym_returns)
        dd = _max_drawdown_from_returns(sym_returns)

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

        metrics.append(
            {
                "simbolo": sym,
                "volatilidad": vol,
                "drawdown": dd,
                "beta": sym_beta,
                "es_benchmark": is_benchmark,
            }
        )

    if not metrics:
        return pd.DataFrame()

    return pd.DataFrame(metrics)


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


class PortfolioViewModelService:
    """Wrapper cacheado alrededor de ``apply_filters``.

    Memoriza el último resultado junto con los totales derivados para evitar
    recomputar mientras no cambien ni las posiciones ni los filtros
    relevantes.
    """

    def __init__(self) -> None:
        self._snapshot: PortfolioViewSnapshot | None = None
        self._dataset_key: str | None = None
        self._filters_key: str | None = None
        self._history_records: list[dict[str, float]] = []

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

        snapshot = PortfolioViewSnapshot(
            df_view=df_view,
            totals=totals,
            apply_elapsed=apply_elapsed,
            totals_elapsed=totals_elapsed,
            generated_at=time.time(),
            historical_total=history_df,
            contribution_metrics=contribution_metrics,
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

