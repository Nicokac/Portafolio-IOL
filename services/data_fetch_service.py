"""Background service responsible for preloading portfolio datasets.

The goal of this module is to decouple the heavy data-fetch pipeline from
Streamlit reruns.  A ``PortfolioDataFetchService`` instance refreshes the
portfolio positions and the quotes required by the viewmodel in the
background and keeps the latest successful snapshot in a shared cache.  The
UI layer and the ``PortfolioViewModelService`` can then reuse the cached
dataset instead of recomputing the whole pipeline on every rerun.

The service is intentionally framework agnostic; it can persist its state in
``CacheService`` (which can be backed by Redis) and exposes explicit methods
to schedule asynchronous refreshes.  The singleton accessor
``get_portfolio_data_fetch_service`` returns a process-wide instance that can
also be wrapped by ``st.cache_resource`` in the Streamlit layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

import pandas as pd

from services.cache import CacheService, fetch_portfolio, fetch_quotes_bulk
from shared.portfolio_utils import unique_symbols

logger = logging.getLogger(__name__)


DatasetBuilder = Callable[[Any, Any], "PortfolioDataset"]


@dataclass
class PortfolioDataset:
    """Normalized snapshot containing the data required by the viewmodel."""

    positions: pd.DataFrame
    quotes: dict[tuple[str, str], Mapping[str, Any]]
    all_symbols: tuple[str, ...]
    available_types: tuple[str, ...]
    dataset_hash: str
    raw_payload: Any | None = None


@dataclass
class DatasetState:
    """Persisted state for the cached dataset."""

    dataset: PortfolioDataset
    updated_at: float
    duration: float
    source: str
    error: str | None = None
    refresh_count: int = 0
    skip_invalidation: bool = False


@dataclass
class DatasetMetadata:
    """Metadata returned together with cached datasets."""

    source: str
    updated_at: float
    stale: bool
    refresh_in_progress: bool
    refresh_count: int
    duration: float
    cache_hit: bool
    error: str | None = None
    skip_invalidation: bool = False


def _compute_dataset_hash(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "empty"
    try:
        hashed = pd.util.hash_pandas_object(df, index=True, categorize=True)
        return hashlib.sha1(hashed.values.tobytes()).hexdigest()
    except TypeError:
        payload = json.dumps(df.to_dict(orient="list"), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()


def _available_types(df: pd.DataFrame) -> tuple[str, ...]:
    if not isinstance(df, pd.DataFrame) or df.empty or "tipo" not in df.columns:
        return ()

    collected: list[str] = []
    seen: set[str] = set()

    for value in df["tipo"].tolist():
        if value in (None, ""):
            continue
        if isinstance(value, float):
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            candidate = str(value)
        elif isinstance(value, str):
            candidate = value
        else:
            candidate = str(value)
        if not candidate:
            continue
        if candidate not in seen:
            collected.append(candidate)
            seen.add(candidate)

    return tuple(sorted(collected))


def _quote_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    if df.empty:
        return []
    cols = [col for col in ("mercado", "simbolo") if col in df.columns]
    if len(cols) < 2:
        return []
    subset = df[cols].dropna(subset=["simbolo"]).astype({"mercado": str, "simbolo": str})
    subset["mercado"] = subset["mercado"].str.lower()
    subset["mercado"] = subset["mercado"].where(subset["mercado"].str.strip().astype(bool), "bcba")
    subset["simbolo"] = subset["simbolo"].str.upper()
    subset = subset.drop_duplicates()
    return list(subset.itertuples(index=False, name=None))


class PortfolioDataFetchService:
    """Manage the lifecycle of the cached portfolio dataset."""

    _CACHE_KEY = "portfolio_dataset_state"

    def __init__(
        self,
        *,
        ttl_seconds: float = 60.0,
        cache: CacheService | None = None,
        builder: DatasetBuilder | None = None,
        time_provider: Callable[[], float] | None = None,
        background_factory: Callable[[Callable[[], None]], threading.Thread] | None = None,
    ) -> None:
        self._ttl_seconds = max(float(ttl_seconds), 1.0)
        self._cache = cache
        self._builder = builder or self._default_builder
        self._time = time_provider or time.time
        self._lock = threading.RLock()
        self._state: DatasetState | None = None
        self._refresh_in_progress = False
        self._refresh_lock = threading.Lock()
        self._background_factory = background_factory
        self._refresh_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def peek_dataset(self) -> tuple[PortfolioDataset | None, DatasetMetadata | None]:
        with self._lock:
            state = self._ensure_state_loaded()
            if state is None:
                return None, None
            metadata = self._build_metadata(state, cache_hit=True)
        return state.dataset, metadata

    def get_dataset(
        self,
        cli: Any,
        psvc: Any,
        *,
        force_refresh: bool = False,
    ) -> tuple[PortfolioDataset, DatasetMetadata]:
        if force_refresh:
            state = self._refresh_dataset(cli, psvc, explicit=True)
            return state.dataset, self._build_metadata(state, cache_hit=False)

        with self._lock:
            state = self._ensure_state_loaded()
            if state is None:
                # Release lock before heavy refresh to avoid blocking readers.
                pass
            else:
                metadata = self._build_metadata(state, cache_hit=True)
                if metadata.stale:
                    self._ensure_background_refresh(cli, psvc)
                return state.dataset, metadata

        state = self._refresh_dataset(cli, psvc, explicit=True)
        return state.dataset, self._build_metadata(state, cache_hit=False)

    def update_quotes(
        self,
        dataset_hash: str,
        quotes: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> None:
        with self._lock:
            state = self._ensure_state_loaded()
            if state is None or state.dataset.dataset_hash != dataset_hash:
                return
            cloned = {tuple(pair): dict(payload) for pair, payload in quotes.items()}
            state.dataset = replace(state.dataset, quotes=cloned)
            state.updated_at = self._time()
            self._store_state(state)

    def schedule_refresh(self, cli: Any, psvc: Any) -> bool:
        return self._ensure_background_refresh(cli, psvc)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _ensure_state_loaded(self) -> DatasetState | None:
        if self._state is not None:
            return self._state
        if self._cache is None:
            return None
        cached = self._cache.get(self._CACHE_KEY)
        if isinstance(cached, DatasetState):
            if not hasattr(cached, "skip_invalidation"):
                cached.skip_invalidation = False
            self._state = cached
        return self._state

    def _build_metadata(
        self,
        state: DatasetState,
        *,
        cache_hit: bool,
    ) -> DatasetMetadata:
        stale = (self._time() - state.updated_at) >= self._ttl_seconds
        return DatasetMetadata(
            source=state.source,
            updated_at=state.updated_at,
            stale=stale,
            refresh_in_progress=self._refresh_in_progress,
            refresh_count=state.refresh_count,
            duration=state.duration,
            cache_hit=cache_hit,
            error=state.error,
            skip_invalidation=state.skip_invalidation,
        )

    def _store_state(self, state: DatasetState) -> None:
        self._state = state
        if self._cache is not None:
            try:
                self._cache.set(self._CACHE_KEY, state, ttl=self._ttl_seconds * 3)
            except Exception:  # pragma: no cover - defensive persistence errors
                logger.debug("No se pudo persistir el dataset en cache", exc_info=True)

    def _ensure_background_refresh(self, cli: Any, psvc: Any) -> bool:
        if self._refresh_in_progress:
            return False
        if self._background_factory is None:
            thread = threading.Thread(
                target=self._run_background_refresh,
                args=(cli, psvc),
                daemon=True,
            )
        else:
            thread = self._background_factory(lambda: self._run_background_refresh(cli, psvc))
        with self._refresh_lock:
            if self._refresh_in_progress:
                return False
            self._refresh_in_progress = True
            self._refresh_thread = thread
            thread.start()
        return True

    def _run_background_refresh(self, cli: Any, psvc: Any) -> None:
        try:
            self._refresh_dataset(cli, psvc, explicit=False)
        except Exception:  # pragma: no cover - background errors are logged
            logger.exception("Refresh en background del dataset falló")
        finally:
            self._refresh_in_progress = False

    def _refresh_dataset(
        self,
        cli: Any,
        psvc: Any,
        *,
        explicit: bool,
    ) -> DatasetState:
        start = self._time()
        dataset = self._builder(cli, psvc)
        duration = self._time() - start
        source = "cache"
        if isinstance(dataset.raw_payload, Mapping):
            cached = bool(dataset.raw_payload.get("_cached"))
            source = "cache" if cached else "api"
        state = DatasetState(
            dataset=dataset,
            updated_at=self._time(),
            duration=duration,
            source=source,
            error=None,
        )
        with self._lock:
            previous = self._ensure_state_loaded()
            if previous is not None:
                state.refresh_count = previous.refresh_count + 1
                previous_hash = getattr(previous.dataset, "dataset_hash", None)
                if previous_hash == dataset.dataset_hash:
                    state.skip_invalidation = True
                    logger.info(
                        'portfolio_data_fetch event="skip_invalidation" dataset_hash=%s',
                        dataset.dataset_hash,
                    )
                else:
                    state.skip_invalidation = False
            self._store_state(state)
        logger.info(
            "Dataset de portafolio actualizado (source=%s, duration=%.3fs, explicit=%s)",
            state.source,
            state.duration,
            explicit,
        )
        return state

    # ------------------------------------------------------------------
    # dataset builder
    # ------------------------------------------------------------------
    def _default_builder(self, cli: Any, psvc: Any) -> PortfolioDataset:
        payload = fetch_portfolio(cli)
        df_pos = psvc.normalize_positions(payload)
        if not isinstance(df_pos, pd.DataFrame):
            df_pos = pd.DataFrame(df_pos)
        df_pos = df_pos.copy()
        dataset_hash = _compute_dataset_hash(df_pos)
        if isinstance(df_pos, pd.DataFrame) and "simbolo" in df_pos.columns:
            symbols = unique_symbols(df_pos["simbolo"])
        else:
            symbols = ()
        types = _available_types(df_pos)
        pairs = _quote_pairs(df_pos)
        quotes = fetch_quotes_bulk(cli, pairs) if pairs else {}
        return PortfolioDataset(
            positions=df_pos,
            quotes={tuple(k): dict(v) for k, v in quotes.items()},
            all_symbols=symbols,
            available_types=types,
            dataset_hash=dataset_hash,
            raw_payload=payload,
        )


_SERVICE_SINGLETON: PortfolioDataFetchService | None = None
_SERVICE_LOCK = threading.Lock()


def get_portfolio_data_fetch_service() -> PortfolioDataFetchService:
    global _SERVICE_SINGLETON
    if _SERVICE_SINGLETON is not None:
        return _SERVICE_SINGLETON
    with _SERVICE_LOCK:
        if _SERVICE_SINGLETON is None:
            _SERVICE_SINGLETON = PortfolioDataFetchService(
                cache=CacheService(namespace="portfolio_dataset_service"),
            )
    return _SERVICE_SINGLETON


__all__ = [
    "DatasetMetadata",
    "DatasetState",
    "PortfolioDataFetchService",
    "PortfolioDataset",
    "get_portfolio_data_fetch_service",
    "_compute_dataset_hash",
]
