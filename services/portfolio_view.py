from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from application.portfolio_service import PortfolioTotals, calculate_totals


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioViewSnapshot:
    """Resultado cacheado del portafolio."""

    df_view: pd.DataFrame
    totals: PortfolioTotals
    apply_elapsed: float
    totals_elapsed: float
    generated_at: float


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

        snapshot = PortfolioViewSnapshot(
            df_view=df_view,
            totals=totals,
            apply_elapsed=apply_elapsed,
            totals_elapsed=totals_elapsed,
            generated_at=time.time(),
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

