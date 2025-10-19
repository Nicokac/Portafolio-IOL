import csv
import hashlib
import json
import logging
import os
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, cast

import streamlit as st
import pandas as pd

from domain.models import Controls
from ui.sidebar_controls import render_sidebar
from ui.fundamentals import render_fundamental_data
from ui.export import PLOTLY_CONFIG
from ui.charts import plot_technical_analysis_chart
from ui.favorites import render_favorite_badges, render_favorite_toggle
from application.portfolio_service import PortfolioService, map_to_us_ticker
from application.ta_service import TAService
from application.portfolio_viewmodel import build_portfolio_viewmodel
from shared import skeletons
from shared.errors import AppError
from shared.favorite_symbols import FavoriteSymbols, get_persistent_favorites
from services.notifications import NotificationFlags, NotificationsService
from services.health import record_tab_latency
from services.portfolio_view import (
    PortfolioViewModelService,
    update_charts,
    update_summary_section,
    update_table_data,
)
from services import snapshots as snapshot_service
from ui.notifications import render_technical_badge, tab_badge_label, tab_badge_suffix
from ui.lazy import charts_fragment, current_component, current_scope, in_form_scope, table_fragment
from ui.lazy.runtime import current_dataset_token
from shared.utils import _as_float_or_none, format_money
from services.performance_metrics import measure_execution
from services.performance_timer import profile_block, record_stage as log_performance_stage
from services.cache import CacheService
from shared.telemetry import log_default_telemetry, log_telemetry
from shared.user_actions import log_user_action
from shared.cache import visual_cache_registry
from infrastructure.iol.auth import get_current_user_id
from shared.fragment_state import (
    FragmentGuardResult,
    fragment_state_soft_refresh,
    get_fragment_state_guardian,
)

from .load_data import load_portfolio_data
from .charts import (
    render_basic_section,
    render_summary as render_summary_section,
    render_table as render_table_section,
    render_charts as render_charts_section,
    render_advanced_analysis,
)
from .fundamentals import render_fundamental_analysis
logger = logging.getLogger(__name__)
session_logger = logging.getLogger("controllers.portfolio.session")

_SERVICE_REGISTRY_KEY = "__portfolio_services__"
_VIEW_MODEL_SERVICE_KEY = "view_model_service"
_VIEW_MODEL_FACTORY_KEY = "view_model_service_factory"
_NOTIFICATIONS_SERVICE_KEY = "notifications_service"
_NOTIFICATIONS_FACTORY_KEY = "notifications_service_factory"
_SNAPSHOT_BACKEND_KEY = "snapshot_backend_override"
_PORTFOLIO_SERVICE_KEY = "portfolio_service"
_TA_SERVICE_KEY = "ta_service"

_CACHE_TTL_SECONDS = 300.0
_INCREMENTAL_CACHE = CacheService(namespace="portfolio_incremental")

_RENDER_REFS_STATE_KEY = "render_refs"

_DATASET_FINGERPRINT_CACHE_KEY = "__portfolio_dataset_fingerprints__"
_DATASET_FINGERPRINT_STATS_KEY = "__portfolio_dataset_fingerprint_stats__"
_MAX_DATASET_FINGERPRINTS = 32
_DATASET_CACHE_FALLBACK: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_DATASET_STATS_FALLBACK: dict[str, Any] = {
    "hits": 0,
    "misses": 0,
    "hit_ratio": 0.0,
    "last_status": None,
    "last_latency_ms": 0.0,
    "last_key": None,
}

_TAB_METRICS_PATH = Path("performance_metrics_14.csv")
_VISUAL_CACHE_STATE_KEY = "cached_render"
_DATASET_HASH_STATE_KEY = "dataset_hash"
_DATASET_REUSE_FLAG_KEY = "__portfolio_visual_cache_reused__"
_PORTFOLIO_LAST_USER_STATE_KEY = "__portfolio_last_user_id__"
_LAZY_BLOCKS_STATE_KEY = "lazy_blocks"
_LAZY_FLAGS_STATE_KEY = "__portfolio_lazy_flag_tokens__"
_VISUAL_CACHE_EVENT_STATE_KEY = "__portfolio_visual_cache_event__"
_UI_PERSIST_STATE_KEY = "portfolio_ui_persist_ms"
_CONTROLS_SNAPSHOT_STATE_KEY = "__portfolio_controls_snapshot__"
_EMPTY_RENDER_LOG_KEY = "__portfolio_empty_render_hash__"
_REFRESH_LOG_STATE_KEY = "__portfolio_last_refresh_hash__"

_VISUAL_STATE_LOCK = threading.RLock()

_RISK_MODULE: Any | None = None


def _load_risk_module() -> Any:
    global _RISK_MODULE
    if _RISK_MODULE is None:
        try:
            from . import risk as risk_module
        except Exception:  # pragma: no cover - defensive import guard
            return None
        _RISK_MODULE = risk_module
    return _RISK_MODULE


def _append_tab_metric(
    tab_name: str,
    duration_s: float,
    *,
    profile_ms: float | None = None,
    overhead_ms: float | None = None,
) -> None:
    """Append a telemetry row for ``tab_name`` to the CSV metrics file."""

    try:
        safe_duration = max(float(duration_s), 0.0)
    except Exception:
        safe_duration = 0.0

    def _format_ms(value: float | None) -> str:
        if value is None:
            return ""
        try:
            return f"{max(float(value), 0.0):.2f}"
        except Exception:
            return ""

    extra = {
        "tab_name": str(tab_name),
        "portfolio_tab_render_s": f"{safe_duration:.6f}",
        "streamlit_overhead_ms": _format_ms(overhead_ms),
        "profile_block_total_ms": _format_ms(profile_ms),
    }
    try:
        log_telemetry(
            (_TAB_METRICS_PATH,),
            phase="portfolio.tab_render",
            elapsed_s=safe_duration,
            extra=extra,
        )
    except Exception:  # pragma: no cover - best effort logging
        logger.debug(
            "No se pudo actualizar %s con la métrica de la pestaña %s",
            _TAB_METRICS_PATH,
            tab_name,
            exc_info=True,
        )


def _get_service_registry() -> dict[str, Any]:
    """Return the per-session registry that stores portfolio services."""

    state = getattr(st, "session_state", None)
    if state is not None:
        try:
            registry = state.get(_SERVICE_REGISTRY_KEY)  # type: ignore[attr-defined]
        except AttributeError:
            try:
                registry = state[_SERVICE_REGISTRY_KEY]  # type: ignore[index]
            except Exception:  # pragma: no cover - defensive branch
                registry = None
        if not isinstance(registry, dict):
            registry = {}
            try:
                state[_SERVICE_REGISTRY_KEY] = registry  # type: ignore[index]
            except Exception:  # pragma: no cover - defensive branch
                pass
        if isinstance(registry, dict):
            return registry

    return {}


def default_view_model_service_factory() -> PortfolioViewModelService:
    """Create a portfolio view model service bound to the configured backend."""

    registry = _get_service_registry()
    snapshot_backend = registry.get(_SNAPSHOT_BACKEND_KEY)
    backend = snapshot_backend if snapshot_backend is not None else snapshot_service
    return PortfolioViewModelService(snapshot_backend=backend)


def default_notifications_service_factory() -> NotificationsService:
    """Return a fresh notifications service instance."""

    return NotificationsService()


def default_portfolio_service_factory() -> PortfolioService:
    """Return a fresh portfolio service instance."""

    return PortfolioService()


def default_ta_service_factory() -> TAService:
    """Return a fresh technical-analysis service instance."""

    return TAService()


def _get_or_create_service(
    key: str,
    *,
    default_factory: Callable[[], Any],
    override_factory: Callable[[], Any] | None = None,
) -> Any:
    """Return a cached service for the current session, creating it if needed."""

    registry = _get_service_registry()
    factory_key = f"{key}_factory"
    if override_factory is not None:
        registry[factory_key] = override_factory
        registry.pop(key, None)

    factory = registry.get(factory_key)
    if not callable(factory):
        factory = default_factory
        registry[factory_key] = factory

    service = registry.get(key)
    if service is None:
        service = factory()
        registry[key] = service
    return service


def get_portfolio_view_service(
    factory: Callable[[], PortfolioViewModelService] | None = None,
) -> PortfolioViewModelService:
    """Return the cached portfolio view service, creating it if necessary."""

    service = _get_or_create_service(
        _VIEW_MODEL_SERVICE_KEY,
        default_factory=default_view_model_service_factory,
        override_factory=factory,
    )
    return cast(PortfolioViewModelService, service)


def get_notifications_service(
    factory: Callable[[], NotificationsService] | None = None,
) -> NotificationsService:
    """Return the cached notifications service, creating it if necessary."""

    service = _get_or_create_service(
        _NOTIFICATIONS_SERVICE_KEY,
        default_factory=default_notifications_service_factory,
        override_factory=factory,
    )
    return cast(NotificationsService, service)


def get_portfolio_service(
    factory: Callable[[], PortfolioService] | None = None,
) -> PortfolioService:
    """Return the cached portfolio service instance."""

    service = _get_or_create_service(
        _PORTFOLIO_SERVICE_KEY,
        default_factory=default_portfolio_service_factory,
        override_factory=factory,
    )
    return cast(PortfolioService, service)


def get_ta_service(factory: Callable[[], TAService] | None = None) -> TAService:
    """Return the cached technical-analysis service instance."""

    service = _get_or_create_service(
        _TA_SERVICE_KEY,
        default_factory=default_ta_service_factory,
        override_factory=factory,
    )
    return cast(TAService, service)


def reset_portfolio_services() -> None:
    """Clear cached portfolio services for the current session."""

    if getattr(st, "session_state", None) is None:
        return

    registry = _get_service_registry()
    for key in (
        _VIEW_MODEL_SERVICE_KEY,
        _VIEW_MODEL_FACTORY_KEY,
        _NOTIFICATIONS_SERVICE_KEY,
        _NOTIFICATIONS_FACTORY_KEY,
        _PORTFOLIO_SERVICE_KEY,
        _TA_SERVICE_KEY,
    ):
        registry.pop(key, None)


def configure_snapshot_backend(snapshot_backend: Any | None) -> None:
    """Override the snapshot backend used by the cached portfolio service."""

    if getattr(st, "session_state", None) is None:
        return

    registry = _get_service_registry()
    registry[_SNAPSHOT_BACKEND_KEY] = snapshot_backend

    service = registry.get(_VIEW_MODEL_SERVICE_KEY)
    configure = getattr(service, "configure_snapshot_backend", None)
    if callable(configure):
        configure(snapshot_backend)


def _apply_tab_badges(tab_labels: list[str], flags: NotificationFlags) -> list[str]:
    """Return updated tab labels including descriptive badge suffixes for active flags."""

    updated = list(tab_labels)

    def _append(label: str, variant: str) -> str:
        suffix = tab_badge_suffix(variant)
        descriptor = tab_badge_label(variant)
        icon = suffix.strip()
        if descriptor in label or (icon and icon in label):
            return label
        annotation = f"{suffix} {descriptor}" if icon else f" ({descriptor})"
        return f"{label}{annotation}"

    if flags.risk_alert and len(updated) > 2:
        updated[2] = _append(updated[2], "risk")
    if flags.upcoming_earnings and len(updated) > 3:
        updated[3] = _append(updated[3], "earnings")
    if flags.technical_signal and len(updated) > 4:
        updated[4] = _append(updated[4], "technical")
    return updated


def _hash_dataframe(df: Any) -> str:
    """Return a stable hash for the provided DataFrame-like object."""

    if not isinstance(df, pd.DataFrame):
        return "none"
    if df.empty:
        return "empty"
    try:
        hashed = pd.util.hash_pandas_object(df, index=True, categorize=True)
        return hashlib.sha1(hashed.values.tobytes()).hexdigest()
    except TypeError:
        payload = df.to_dict(orient="list")
    except Exception:
        payload = json.loads(df.to_json(orient="split", date_format="iso"))
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(data).hexdigest()


def _current_dataset_hash() -> str:
    """Return the dataset hash persisted in the session, if any."""

    try:
        state = getattr(st, "session_state", None)
    except Exception:  # pragma: no cover - defensive safeguard
        return ""
    if state is None:
        return ""
    try:
        dataset_hash = state.get(_DATASET_HASH_STATE_KEY)
    except Exception:  # pragma: no cover - defensive safeguard
        dataset_hash = None
    if isinstance(dataset_hash, str):
        return dataset_hash
    return ""


def _get_dataset_fingerprint_cache() -> "OrderedDict[str, dict[str, Any]]":
    """Return the in-session cache for portfolio dataset fingerprints."""

    state = getattr(st, "session_state", None)
    cache: "OrderedDict[str, dict[str, Any]]" | None = None
    if state is not None:
        raw_cache = state.get(_DATASET_FINGERPRINT_CACHE_KEY)
        if isinstance(raw_cache, OrderedDict):
            cache = raw_cache
        elif isinstance(raw_cache, dict):
            cache = OrderedDict(raw_cache)
        else:
            cache = OrderedDict()
        try:
            state[_DATASET_FINGERPRINT_CACHE_KEY] = cache
        except Exception:  # pragma: no cover - defensive safeguard
            pass
    if cache is None:
        cache = _DATASET_CACHE_FALLBACK
    return cache


def _get_dataset_fingerprint_stats() -> dict[str, Any]:
    """Return mutable statistics for the dataset fingerprint cache."""

    state = getattr(st, "session_state", None)
    stats: dict[str, Any] | None = None
    if state is not None:
        raw_stats = state.get(_DATASET_FINGERPRINT_STATS_KEY)
        if isinstance(raw_stats, dict):
            stats = raw_stats
        else:
            stats = dict(_DATASET_STATS_FALLBACK)
        try:
            state[_DATASET_FINGERPRINT_STATS_KEY] = stats
        except Exception:  # pragma: no cover - defensive safeguard
            pass
    if stats is None:
        stats = _DATASET_STATS_FALLBACK
    return stats


def _publish_fingerprint_stats(stats: Mapping[str, Any]) -> None:
    """Expose the latest fingerprint cache stats for diagnostics."""

    snapshot = {
        "hits": int(stats.get("hits", 0) or 0),
        "misses": int(stats.get("misses", 0) or 0),
        "hit_ratio": float(stats.get("hit_ratio", 0.0) or 0.0),
        "last_status": stats.get("last_status"),
        "last_latency_ms": float(stats.get("last_latency_ms", 0.0) or 0.0),
        "last_key": stats.get("last_key"),
    }
    try:
        st.session_state["portfolio_fingerprint_cache_stats"] = snapshot
    except Exception:  # pragma: no cover - defensive safeguard
        pass


def _record_fingerprint_event(status: str, elapsed_ms: float, cache_key: str) -> None:
    """Update statistics and telemetry for fingerprint cache activity."""

    stats = _get_dataset_fingerprint_stats()
    bucket = "hits" if status == "hit" else "misses"
    stats[bucket] = int(stats.get(bucket, 0) or 0) + 1
    total = int(stats.get("hits", 0) or 0) + int(stats.get("misses", 0) or 0)
    stats["hit_ratio"] = float(stats.get("hits", 0) or 0) / total if total else 0.0
    stats["last_status"] = status
    stats["last_latency_ms"] = float(elapsed_ms)
    stats["last_key"] = cache_key
    try:
        st.session_state[_DATASET_FINGERPRINT_STATS_KEY] = stats
    except Exception:  # pragma: no cover - defensive safeguard
        pass
    _publish_fingerprint_stats(stats)
    try:
        log_performance_stage(
            "portfolio_ui.fingerprint_cache",
            total_ms=elapsed_ms,
            extra={"status": status},
        )
    except Exception:  # pragma: no cover - telemetry should not break rendering
        logger.debug(
            "No se pudo registrar métricas de fingerprint cache", exc_info=True
        )


def _publish_visual_cache_debug() -> None:
    """Expose visual cache diagnostics when developer tooling is enabled."""

    if os.getenv("ST_DEBUG_CACHE") != "1":
        return
    try:
        st.session_state["portfolio_visual_cache_debug"] = visual_cache_registry.snapshot()
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug(
            "No se pudo exponer el estado de depuración de la caché visual", exc_info=True
        )


def _dataset_filters_key(controls: Any) -> str:
    """Return a hash describing the filters that affect the dataset view."""

    payload = {
        "hide_cash": getattr(controls, "hide_cash", None),
        "selected_syms": sorted(
            str(sym) for sym in getattr(controls, "selected_syms", []) or ()
        ),
        "selected_types": sorted(
            str(tp) for tp in getattr(controls, "selected_types", []) or ()
        ),
        "symbol_query": (getattr(controls, "symbol_query", "") or "").strip(),
        "order_by": getattr(controls, "order_by", None),
        "desc": getattr(controls, "desc", None),
        "show_usd": getattr(controls, "show_usd", None),
        "top_n": getattr(controls, "top_n", None),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _favorites_signature(favorites: Any) -> tuple[str, ...]:
    """Return a deterministic tuple describing the current favorites list."""

    if favorites is None:
        return ()
    getter = getattr(favorites, "list", None)
    if not callable(getter):
        return ()
    try:
        return tuple(sorted(str(sym) for sym in getter()))
    except Exception:
        logger.debug("No se pudo obtener la lista de favoritos", exc_info=True)
        return ()


def _compute_portfolio_dataset_key(viewmodel: Any, df_view: Any) -> str:
    """Compute the dataset fingerprint without consulting memoized values."""

    snapshot_id = getattr(viewmodel, "snapshot_id", None)
    if snapshot_id:
        return f"id:{snapshot_id}"

    totals = getattr(viewmodel, "totals", None)
    totals_sig = "|".join(
        str(getattr(totals, attr, ""))
        for attr in ("total_value", "total_cost", "total_pl", "total_pl_pct")
    )
    history = _hash_dataframe(getattr(viewmodel, "historical_total", None))
    contributions = getattr(viewmodel, "contributions", None)
    contrib_sig = ""
    if contributions is not None:
        contrib_sig = "|".join(
            [
                _hash_dataframe(getattr(contributions, "by_symbol", None)),
                _hash_dataframe(getattr(contributions, "by_type", None)),
            ]
        )

    return "|".join(
        [
            _hash_dataframe(df_view),
            history,
            contrib_sig,
            totals_sig,
        ]
    )


def _portfolio_dataset_key(viewmodel: Any, df_view: Any) -> str:
    """Return a fingerprint for the current portfolio dataset with memoization."""

    controls = getattr(viewmodel, "controls", None)
    filters_hash = _dataset_filters_key(controls)
    snapshot_id = getattr(viewmodel, "snapshot_id", None)
    if snapshot_id:
        base_key = f"id:{snapshot_id}"
    else:
        base_key = f"obj:{id(df_view)}"
    cache_key = f"{base_key}|{filters_hash}"

    cache = _get_dataset_fingerprint_cache()
    status = "hit"
    start = time.perf_counter()
    with measure_execution("portfolio_ui.fingerprint_cache"):
        entry = cache.get(cache_key)
        fingerprint = entry.get("value") if isinstance(entry, Mapping) else None
        if not isinstance(fingerprint, str):
            status = "miss"
            fingerprint = _compute_portfolio_dataset_key(viewmodel, df_view)
            cache[cache_key] = {
                "value": fingerprint,
                "timestamp": time.time(),
                "snapshot": snapshot_id,
                "filters": filters_hash,
            }
            while len(cache) > _MAX_DATASET_FINGERPRINTS:
                cache.popitem(last=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _record_fingerprint_event(status, elapsed_ms, cache_key)
    return fingerprint


def _summary_filters_key(controls: Any, metrics: Any, favorites: Any, snapshot: Any) -> str:
    """Return a hashable representation of summary-relevant filters."""

    payload = {
        "hide_cash": getattr(controls, "hide_cash", None),
        "selected_syms": sorted(str(sym) for sym in getattr(controls, "selected_syms", []) or ()),
        "selected_types": sorted(str(tp) for tp in getattr(controls, "selected_types", []) or ()),
        "symbol_query": (getattr(controls, "symbol_query", "") or "").strip(),
        "ccl_rate": getattr(metrics, "ccl_rate", None),
        "snapshot": getattr(snapshot, "storage_id", None) or getattr(snapshot, "id", None),
        "favorites": _favorites_signature(favorites),
        "pending": ";".join(
            sorted(str(item) for item in getattr(snapshot, "pending_metrics", ()) or ())
        ),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _table_filters_key(controls: Any, favorites: Any) -> str:
    """Return a hash representing table-specific filters."""

    payload = {
        "order_by": getattr(controls, "order_by", None),
        "desc": getattr(controls, "desc", None),
        "show_usd": getattr(controls, "show_usd", None),
        "favorites": _favorites_signature(favorites),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _charts_filters_key(controls: Any) -> str:
    """Return a hash for chart configuration filters."""

    payload = {
        "top_n": getattr(controls, "top_n", None),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _component_cache_key(
    portfolio_id: str,
    filters_hash: str,
    tab_slug: str,
    component: str,
) -> str:
    slug = tab_slug or "tab"
    return f"{portfolio_id}|{filters_hash}|{slug}|{component}"


def _get_component_metadata(
    portfolio_id: str,
    filters_hash: str,
    tab_slug: str,
    component: str,
) -> Mapping[str, float] | None:
    """Return cached metadata for the given component if available."""

    key = _component_cache_key(portfolio_id, filters_hash, tab_slug, component)
    cached = _INCREMENTAL_CACHE.get(key)
    if isinstance(cached, Mapping):
        return cached
    return None


def _store_component_metadata(
    portfolio_id: str,
    filters_hash: str,
    tab_slug: str,
    component: str,
    computed_at: float,
) -> None:
    """Persist metadata for a rendered component with TTL."""

    key = _component_cache_key(portfolio_id, filters_hash, tab_slug, component)
    payload = {"computed_at": float(computed_at)}
    _INCREMENTAL_CACHE.set(key, payload, ttl=_CACHE_TTL_SECONDS)


def _ensure_component_store(tab_cache: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Return the component cache store associated with the current tab."""

    if tab_cache is None:
        return {}
    store = tab_cache.get("components")
    if not isinstance(store, dict):
        store = {}
        tab_cache["components"] = store
    return store


def _ensure_component_entry(
    store: dict[str, dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    """Ensure component scaffolding exists for ``name`` and return it."""

    entry = store.get(name)
    if not isinstance(entry, dict):
        entry = {}
        store[name] = entry
    placeholder = entry.get("placeholder")
    if not hasattr(placeholder, "container"):
        placeholder = st.empty()
        entry["placeholder"] = placeholder

    if name in {"table", "charts"}:
        container = entry.get("container")
        if not hasattr(container, "empty"):
            try:
                container = placeholder.container()
            except Exception:  # pragma: no cover - defensive safeguard
                container = st.container()
            entry["container"] = container

        trigger_placeholder = entry.get("trigger_placeholder")
        if not hasattr(trigger_placeholder, "container"):
            try:
                trigger_placeholder = container.empty()
            except Exception:  # pragma: no cover - defensive safeguard
                trigger_placeholder = st.empty()
            entry["trigger_placeholder"] = trigger_placeholder

        body_placeholder = entry.get("body_placeholder")
        if not hasattr(body_placeholder, "container"):
            try:
                body_placeholder = container.empty()
            except Exception:  # pragma: no cover - defensive safeguard
                body_placeholder = st.empty()
            entry["body_placeholder"] = body_placeholder
            entry["placeholder"] = body_placeholder

    return entry


def _should_reset_rendered_flag(
    entry_dataset: str | None,
    dataset_token: str,
    status: str | None,
    *,
    soft_refresh_guard: bool = False,
) -> bool:
    """Return whether the component render flag should reset for this dataset."""

    if soft_refresh_guard:
        return False
    if not dataset_token:
        return False
    if entry_dataset != dataset_token:
        return False
    return status != "loaded"


def _ensure_render_refs() -> dict[str, Any]:
    """Return the persistent placeholder references stored in session state."""

    with _VISUAL_STATE_LOCK:
        refs = st.session_state.get(_RENDER_REFS_STATE_KEY)
        if not isinstance(refs, dict):
            refs = {}
            try:
                st.session_state[_RENDER_REFS_STATE_KEY] = refs
            except Exception:  # pragma: no cover - defensive safeguard
                pass
        return refs


def _get_lazy_flag_store() -> dict[str, Any]:
    """Return the dataset map that tracks persisted lazy triggers."""

    with _VISUAL_STATE_LOCK:
        try:
            store = st.session_state.get(_LAZY_FLAGS_STATE_KEY)
        except Exception:  # pragma: no cover - defensive safeguard
            store = None

        if not isinstance(store, dict):
            store = {}
            try:
                st.session_state[_LAZY_FLAGS_STATE_KEY] = store
            except Exception:  # pragma: no cover - defensive safeguard
                pass
        return store


def _mark_lazy_flag_ready(key: str | None, dataset_token: str) -> None:
    """Persist that ``key`` has been triggered for ``dataset_token``."""

    if not key:
        return

    with _VISUAL_STATE_LOCK:
        try:
            st.session_state[key] = True
        except Exception:  # pragma: no cover - defensive safeguard
            pass

        store = _get_lazy_flag_store()
        store[key] = {"dataset": dataset_token, "ts": time.time()}


def _clear_lazy_flag(key: str | None) -> None:
    """Reset the persistent flag associated with a lazy block."""

    if not key:
        return

    with _VISUAL_STATE_LOCK:
        try:
            st.session_state[key] = False
        except Exception:  # pragma: no cover - defensive safeguard
            try:
                st.session_state.pop(key, None)
            except Exception:
                pass

        store = st.session_state.get(_LAZY_FLAGS_STATE_KEY)
        if isinstance(store, dict):
            store.pop(key, None)


def _lazy_flag_ready(key: str | None, dataset_token: str) -> bool:
    """Return whether ``key`` should unlock the lazy block for ``dataset_token``."""

    if not key:
        return False

    with _VISUAL_STATE_LOCK:
        try:
            value = st.session_state.get(key)
        except Exception:  # pragma: no cover - defensive safeguard
            value = None

        store = st.session_state.get(_LAZY_FLAGS_STATE_KEY)
        if isinstance(store, dict):
            record = store.get(key)
            if isinstance(record, dict):
                stored_dataset = record.get("dataset")
                if stored_dataset and stored_dataset != dataset_token:
                    store.pop(key, None)
                    try:
                        st.session_state.pop(key, None)
                    except Exception:  # pragma: no cover - defensive safeguard
                        pass
                    return False

        return bool(value)


def _ensure_lazy_blocks(dataset_token: str) -> dict[str, dict[str, Any]]:
    """Ensure the lazy loading scaffolding exists for the active dataset."""

    with _VISUAL_STATE_LOCK:
        state = st.session_state.get(_LAZY_BLOCKS_STATE_KEY)
        if not isinstance(state, dict):
            state = {}
            try:
                st.session_state[_LAZY_BLOCKS_STATE_KEY] = state
            except Exception:  # pragma: no cover - defensive safeguard
                pass

        lazy_blocks: dict[str, dict[str, Any]] = {}
        for name in ("table", "charts"):
            block = state.get(name)
            if not isinstance(block, dict) or block.get("dataset_hash") != dataset_token:
                block = {
                    "status": "pending",
                    "dataset_hash": dataset_token,
                    "triggered_at": None,
                    "loaded_at": None,
                    "prompt_rendered": False,
                    "lazy_load_ms": None,
                    "mount_latency_ms": None,
                }
                state[name] = block
            else:
                if block.get("status") not in ("pending", "loaded"):
                    block["status"] = "pending"
                block.setdefault("dataset_hash", dataset_token)
                block.setdefault("triggered_at", None)
                block.setdefault("loaded_at", None)
                block.setdefault("prompt_rendered", False)
                block.setdefault("lazy_load_ms", None)
                block.setdefault("mount_latency_ms", None)
            lazy_blocks[name] = block
        return lazy_blocks


def _record_ui_persist_visibility(block: dict[str, Any], *, visible: bool) -> None:
    """Track how long lazy UI elements remain visible without flicker."""

    if not isinstance(block, dict):
        return

    with _VISUAL_STATE_LOCK:
        if not visible:
            block.pop("_ui_persist_start", None)
            try:
                st.session_state.pop(_UI_PERSIST_STATE_KEY, None)
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug("No se pudo limpiar ui_persist_ms", exc_info=True)
            return

        now = time.time()
        loaded_at = block.get("loaded_at")
        start = None
        if isinstance(loaded_at, (int, float)):
            start = float(loaded_at)
        else:
            start = block.get("_ui_persist_start")
            if not isinstance(start, (int, float)):
                start = now
                block["_ui_persist_start"] = start

        try:
            persist_ms = max((now - float(start)) * 1000.0, 0.0)
        except Exception:  # pragma: no cover - defensive safeguard
            persist_ms = 0.0

        try:
            existing = st.session_state.get(_UI_PERSIST_STATE_KEY)
        except Exception:  # pragma: no cover - defensive safeguard
            existing = None
        if isinstance(existing, (int, float)):
            persist_ms = max(persist_ms, float(existing))

        persist_ms = round(float(persist_ms), 2)
        try:
            st.session_state[_UI_PERSIST_STATE_KEY] = persist_ms
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug("No se pudo persistir ui_persist_ms", exc_info=True)

def _render_lazy_trigger(placeholder: Any, *, label: str, session_key: str | None) -> bool:
    """Render a persistent widget that toggles the lazy loading flag."""

    if not session_key:
        return False

    try:
        current_value = bool(st.session_state.get(session_key))
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo verificar el estado diferido %s", session_key, exc_info=True)
        current_value = False

    if in_form_scope():
        submit_callable = getattr(st, "form_submit_button", None)
        if callable(submit_callable):
            try:
                pressed = bool(submit_callable(label, key=session_key))
            except TypeError:
                pressed = bool(submit_callable(label))
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug(
                    "No se pudo renderizar el form submit para %s",
                    session_key,
                    exc_info=True,
                )
            else:
                if pressed:
                    try:
                        st.session_state[session_key] = True
                    except Exception:  # pragma: no cover - defensive safeguard
                        logger.debug(
                            "No se pudo persistir la llave diferida %s",
                            session_key,
                            exc_info=True,
                        )
                return pressed or current_value

    widget_callable = None
    for attr in ("toggle", "checkbox"):
        widget_callable = getattr(placeholder, attr, None)
        if callable(widget_callable):
            break
    if not callable(widget_callable):
        for attr in ("toggle", "checkbox"):
            widget_callable = getattr(st, attr, None)
            if callable(widget_callable):
                break
    result = current_value
    if callable(widget_callable):
        try:
            result = bool(widget_callable(label, key=session_key))
        except TypeError:
            try:
                result = bool(
                    widget_callable(label, key=session_key, value=current_value)
                )
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug("No se pudo renderizar el control diferido %s", session_key, exc_info=True)
                result = current_value
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug("No se pudo renderizar el control diferido %s", session_key, exc_info=True)
            result = current_value
    else:
        button_callable = getattr(placeholder, "button", None)
        if not callable(button_callable):
            button_callable = getattr(st, "button", None)
        if callable(button_callable):
            try:
                result = bool(button_callable(label, key=session_key))
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug("No se pudo renderizar el botón diferido %s", session_key, exc_info=True)
                result = current_value
            else:
                if result:
                    try:
                        st.session_state[session_key] = True
                    except Exception:  # pragma: no cover - defensive safeguard
                        logger.debug("No se pudo persistir la llave diferida %s", session_key, exc_info=True)

    try:
        state_value = st.session_state.get(session_key)
    except Exception:  # pragma: no cover - defensive safeguard
        state_value = result
    else:
        if isinstance(state_value, bool):
            result = state_value

    return bool(result)


def _prompt_lazy_block(
    block: dict[str, Any],
    *,
    placeholder: Any,
    button_label: str,
    info_message: str,
    key: str,
    dataset_token: str,
    fallback_key: str | None = None,
) -> bool:
    """Render a persistent lazy trigger returning readiness."""

    guardian = get_fragment_state_guardian()
    session_key = fallback_key or key
    primary_ready = _lazy_flag_ready(key, dataset_token)
    fallback_ready = False
    if fallback_key is not None:
        fallback_ready = _lazy_flag_ready(fallback_key, dataset_token)
    session_ready = primary_ready or fallback_ready

    ready = bool(session_ready)
    scope = current_scope()
    component = current_component() or ("charts" if "chart" in (key or "") else "table")
    resolved_dataset = dataset_token or current_dataset_token() or _current_dataset_hash()
    dataset_hash = str(resolved_dataset or "")
    was_loaded = block.get("status") == "loaded"
    last_scope = block.get("__user_action_scope__")
    if scope != last_scope:
        block["__user_action_scope__"] = scope
        log_user_action(
            "scope_change",
            {"key": key, "scope": scope or "global"},
            dataset_hash=resolved_dataset,
        )
    if not ready and scope == "global":
        ready = True
        block.setdefault("auto_loaded", True)

    guard_result = FragmentGuardResult(rehydrated=False, explicit_hide=False)
    if not ready:
        guard_result = guardian.maybe_rehydrate(
            key=key,
            session_key=session_key,
            dataset_hash=dataset_hash,
            component=component,
            scope=scope,
            was_loaded=was_loaded,
            fallback_key=fallback_key,
        )
        if guard_result.rehydrated:
            ready = True
            session_ready = True

    if info_message and not ready and not block.get("prompt_rendered"):
        try:
            placeholder.write(info_message)
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug("No se pudo mostrar el mensaje diferido %s", key, exc_info=True)
        else:
            block["prompt_rendered"] = True

    if session_key:
        try:
            st.session_state.setdefault(session_key, False)
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug("No se pudo inicializar la llave diferida %s", session_key, exc_info=True)

    trigger_state = False
    has_loaded_flag = block.get("status") == "loaded" or "loaded_at" in block
    should_render_trigger = not ready or block.get("auto_loaded") is True or has_loaded_flag
    if should_render_trigger:
        trigger_state = _render_lazy_trigger(
            placeholder, label=button_label, session_key=session_key
        )
        if trigger_state and not session_ready:
            action = "lazy_block_trigger"
            if "load_table" in (key or ""):
                action = "load_portfolio_table"
            elif "load_chart" in (key or ""):
                action = "load_portfolio_charts"
            detail: dict[str, Any] = {"key": key, "label": button_label}
            if fallback_key:
                detail["fallback_key"] = fallback_key
            log_user_action(action, detail, dataset_hash=resolved_dataset)
        ready = ready or trigger_state

    explicit_hide_during_run = False
    if (
        primary_ready
        and not trigger_state
        and block.get("status") == "loaded"
    ):
        current_flag = False
        if session_key:
            try:
                current_flag = bool(st.session_state.get(session_key))
            except Exception:  # pragma: no cover - defensive safeguard
                current_flag = False
        if current_flag is False:
            _clear_lazy_flag(key)
            primary_ready = False
            fallback_ready = False
            session_ready = False
            ready = False
            explicit_hide_during_run = True

    if ready:
        if block.get("status") != "loaded":
            block["status"] = "loaded"
            if block.get("triggered_at") is None:
                block["triggered_at"] = time.perf_counter()
        if session_key:
            try:
                st.session_state[session_key] = True
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug("No se pudo persistir la llave diferida %s", session_key, exc_info=True)
        _mark_lazy_flag_ready(key, dataset_token)
        _mark_lazy_flag_ready(fallback_key, dataset_token)
        _record_ui_persist_visibility(block, visible=True)
        guardian.mark_ready(
            key=key,
            session_key=session_key,
            dataset_hash=dataset_hash,
            component=component,
            scope=scope,
            fallback_key=fallback_key,
        )
        return True

    block["status"] = "pending"
    block["prompt_rendered"] = False
    _record_ui_persist_visibility(block, visible=False)
    explicit_hide = guard_result.explicit_hide or explicit_hide_during_run
    if explicit_hide:
        _clear_lazy_flag(key)
        _clear_lazy_flag(session_key)
        _clear_lazy_flag(fallback_key)
    guardian.mark_not_ready(
        key=key,
        session_key=session_key,
        dataset_hash=dataset_hash,
        explicit_hide=explicit_hide,
    )
    if resolved_dataset:
        marker = f"{resolved_dataset}:{key}"
        if block.get("__user_action_not_loaded") != marker:
            block["__user_action_not_loaded"] = marker
            log_user_action(
                "lazy_block_not_loaded",
                {"key": key, "label": button_label},
                dataset_hash=resolved_dataset,
            )
    return False


def _record_lazy_component_load(
    component: str,
    elapsed_ms: float,
    dataset_token: str | None,
    *,
    mount_latency_ms: float | None = None,
) -> None:
    """Persist telemetry for a deferred component once it renders."""

    if not dataset_token:
        dataset_token = current_dataset_token() or "none"
    try:
        payload = {
            "lazy_loaded_component": component,
            "lazy_visual_load_ms": max(float(elapsed_ms), 0.0),
        }
        if mount_latency_ms is not None:
            payload["visual_mount_latency_ms"] = max(float(mount_latency_ms), 0.0)
        log_default_telemetry(
            phase="portfolio.lazy_component",
            elapsed_s=max(float(elapsed_ms), 0.0) / 1000.0,
            dataset_hash=dataset_token,
            extra=payload,
        )
    except Exception:  # pragma: no cover - best-effort logging
        logger.debug(
            "No se pudo registrar la telemetría diferida para %s", component, exc_info=True
        )


def _log_quotes_refresh_event(dataset_hash: str | None, *, source: str, detail: Mapping[str, Any] | None = None) -> None:
    """Persist a user action entry describing a quotes refresh."""

    payload: dict[str, Any] = {"source": source}
    if detail:
        payload.update(detail)
    log_user_action("quotes_refresh", payload, dataset_hash=str(dataset_hash or ""))


def _render_updated_caption(timestamp: float | None) -> None:
    """Render a caption with the last update timestamp."""

    if timestamp is None:
        label = "Actualizado: –"
    else:
        dt = datetime.fromtimestamp(timestamp)
        label = f"Actualizado: {dt.strftime('%Y-%m-%d %H:%M:%S')}"
    if _CACHE_TTL_SECONDS:
        minutes = int(_CACHE_TTL_SECONDS // 60)
        label = f"{label} · Caché {minutes} min"
    try:
        st.caption(label)
    except Exception:  # pragma: no cover - defensive
        logger.debug("No se pudo renderizar la marca de tiempo de actualización", exc_info=True)


def render_basic_tab(
    viewmodel,
    favorites,
    snapshot,
    *,
    tab_slug: str = "portafolio",
    tab_cache: dict[str, Any] | None = None,
    timings: dict[str, float] | None = None,
    lazy_metrics: bool = False,
) -> None:
    """Render the summary view for the basic portfolio tab with incremental caching."""

    favorites = favorites or _get_cached_favorites()
    controls = getattr(viewmodel, "controls", Controls())
    metrics = getattr(viewmodel, "metrics", None)
    df_view = getattr(viewmodel, "positions", pd.DataFrame())
    totals = getattr(viewmodel, "totals", None)
    historical_total = getattr(viewmodel, "historical_total", None)
    contributions = getattr(viewmodel, "contributions", None)
    ccl_rate = getattr(metrics, "ccl_rate", None)
    pending_metrics = tuple(getattr(viewmodel, "pending_metrics", ()) or ())
    pending_extended = "extended_metrics" in pending_metrics
    soft_refresh_guard = bool(getattr(snapshot, "soft_refresh_guard", False))

    if lazy_metrics and pending_extended:
        with st.spinner("Calculando métricas extendidas del portafolio..."):
            time.sleep(0.05)
        st.caption("⏳ Calculando métricas extendidas del portafolio...")

    portfolio_id = _portfolio_dataset_key(viewmodel, df_view)
    summary_filters = _summary_filters_key(controls, metrics, favorites, snapshot)
    table_filters = _table_filters_key(controls, favorites)
    chart_filters = _charts_filters_key(controls)

    component_store = _ensure_component_store(tab_cache)
    summary_entry = _ensure_component_entry(component_store, "summary")
    table_entry = _ensure_component_entry(component_store, "table")
    charts_entry = _ensure_component_entry(component_store, "charts")

    render_refs = _ensure_render_refs()
    summary_refs = render_refs.setdefault("summary", {})
    table_refs = render_refs.setdefault("table", {})
    charts_refs = render_refs.setdefault("charts", {})

    table_container = table_entry.get("container")
    if hasattr(table_container, "empty"):
        table_refs.setdefault("container", table_container)

    dataset_hash = st.session_state.get(_DATASET_HASH_STATE_KEY)
    visual_cache_entry = _get_visual_cache_entry(dataset_hash)
    dataset_token = str(dataset_hash or "none")

    lazy_blocks = _ensure_lazy_blocks(dataset_token)
    table_lazy = lazy_blocks["table"]
    charts_lazy = lazy_blocks["charts"]
    table_lazy["placeholder"] = table_entry.get("trigger_placeholder", table_entry.get("placeholder"))
    charts_lazy["placeholder"] = charts_entry.get("trigger_placeholder", charts_entry.get("placeholder"))

    summary_signature = (portfolio_id, summary_filters)
    summary_meta = _get_component_metadata(portfolio_id, summary_filters, tab_slug, "summary")
    summary_entry_hash = summary_entry.get("dataset_hash")
    summary_entry.setdefault("dataset_hash", summary_entry_hash or dataset_token)

    summary_refs["placeholder"] = summary_entry["placeholder"]

    has_positions = bool(summary_entry.get("has_positions", not getattr(df_view, "empty", True)))
    summary_timestamp = summary_entry.get("updated_at")
    previously_rendered_summary = bool(summary_entry.get("rendered"))
    partial_update_start = time.perf_counter()
    should_render_summary = False
    with _record_stage("render_summary", timings):
        should_render_summary = (
            summary_entry.get("dataset_hash") != dataset_token
            or not summary_entry.get("rendered")
        )
        if should_render_summary:
            placeholder = summary_entry["placeholder"]
            references = summary_refs.get("references")
            updated_refs = update_summary_section(
                placeholder,
                render_fn=render_summary_section,
                df_view=df_view,
                controls=controls,
                ccl_rate=ccl_rate,
                totals=totals,
                favorites=favorites,
                historical_total=None if pending_extended else historical_total,
                contribution_metrics=None if pending_extended else contributions,
                snapshot=snapshot,
                references=references,
            )
            summary_refs["references"] = updated_refs
            has_positions = bool(updated_refs.get("has_positions", has_positions))
            summary_timestamp = time.time()
            _render_updated_caption(summary_timestamp)
            summary_entry["signature"] = summary_signature
            summary_entry["rendered"] = True
            summary_entry["has_positions"] = has_positions
            summary_entry["updated_at"] = summary_timestamp
            summary_entry["dataset_hash"] = dataset_token
            summary_refs["dataset_hash"] = dataset_token
            with _VISUAL_STATE_LOCK:
                visual_cache_entry["summary_placeholder"] = placeholder
                visual_cache_entry["summary_rendered"] = True
                visual_cache_entry["summary_timestamp"] = summary_timestamp
            _store_component_metadata(
                portfolio_id,
                summary_filters,
                tab_slug,
                "summary",
                summary_timestamp,
            )
        else:
            has_positions = bool(summary_entry.get("has_positions", has_positions))
            summary_refs.setdefault("dataset_hash", dataset_token)
            summary_entry.setdefault("signature", summary_signature)
            if summary_timestamp is None and summary_meta is not None:
                summary_timestamp = summary_meta.get("computed_at")
                summary_entry.setdefault("updated_at", summary_timestamp)
            with _VISUAL_STATE_LOCK:
                visual_cache_entry.setdefault(
                    "summary_placeholder", summary_entry.get("placeholder")
                )
                visual_cache_entry.setdefault("summary_rendered", True)
                if summary_timestamp is not None:
                    visual_cache_entry.setdefault("summary_timestamp", summary_timestamp)

    summary_reused = (
        bool(summary_entry.get("rendered"))
        and summary_entry.get("dataset_hash") == dataset_token
        and summary_entry.get("signature") == summary_signature
    )
    visual_cache_registry.record(
        "summary",
        dataset_hash=dataset_hash,
        reused=summary_reused,
        signature=summary_signature,
    )

    table_signature = (portfolio_id, table_filters)
    table_meta = _get_component_metadata(portfolio_id, table_filters, tab_slug, "table")
    table_entry_hash = table_entry.get("dataset_hash")
    table_entry.setdefault("dataset_hash", table_entry_hash or dataset_token)
    if soft_refresh_guard:
        table_entry["dataset_hash"] = dataset_token
        table_entry["rendered"] = True
    table_placeholder = table_entry.get("body_placeholder") or table_entry["placeholder"]
    table_trigger_placeholder = table_entry.get("trigger_placeholder") or table_entry["placeholder"]
    previously_rendered_table = bool(table_entry.get("rendered"))
    if soft_refresh_guard:
        previously_rendered_table = True
    table_entry.setdefault("signature", table_signature)
    if not previously_rendered_table and not table_entry.get("skeleton_displayed"):
        skeletons.mark_placeholder("table", placeholder=table_placeholder)
        try:
            table_placeholder.write("⏳ Cargando tabla…")
        except Exception:  # pragma: no cover - defensive guard for Streamlit stubs
            logger.debug("No se pudo mostrar el placeholder inicial de la tabla", exc_info=True)
        else:
            table_entry["skeleton_displayed"] = True
    table_refs["placeholder"] = table_placeholder
    table_entry_dataset = table_entry.get("dataset_hash")
    if (
        has_positions
        and table_entry_dataset == dataset_token
        and table_entry.get("rendered")
    ):
        if table_lazy.get("status") != "loaded":
            table_lazy["status"] = "loaded"
    if _should_reset_rendered_flag(
        table_entry_dataset,
        dataset_token,
        table_lazy.get("status"),
        soft_refresh_guard=soft_refresh_guard,
    ):
        table_entry["rendered"] = False
        previously_rendered_table = False

    with table_fragment(dataset_token=dataset_token) as table_ctx:
        table_ready = _prompt_lazy_block(
            table_lazy,
            placeholder=table_trigger_placeholder,
            button_label="📊 Cargar tabla del portafolio",
            info_message="La tabla principal se cargará cuando la solicites.",
            key=f"{tab_slug}_load_table",
            dataset_token=dataset_token,
            fallback_key="load_table",
        )

        should_render_table = False
        if table_ready:
            with _record_stage("render_table", timings):
                should_render_table = (
                    table_entry.get("dataset_hash") != dataset_token
                    or not table_entry.get("rendered")
                )
                if should_render_table:
                    placeholder = table_placeholder
                    references = table_refs.get("references")
                    start_trigger = table_lazy.get("triggered_at")
                    spinner_factory = getattr(st, "spinner", None)
                    spinner_cm = (
                        spinner_factory("Procesando tabla del portafolio…")
                        if callable(spinner_factory)
                        else nullcontext()
                    )
                    mount_start = time.perf_counter()
                    with spinner_cm:
                        updated_refs = update_table_data(
                            placeholder,
                            render_fn=render_table_section,
                            df_view=df_view,
                            controls=controls,
                            ccl_rate=ccl_rate,
                            favorites=favorites,
                            references=references,
                        )
                    mount_end = time.perf_counter()
                    table_refs["references"] = updated_refs
                    table_timestamp = time.time()
                    mount_latency_ms = max((mount_end - mount_start) * 1000.0, 0.0)
                    elapsed_ms = mount_latency_ms
                    if isinstance(start_trigger, (int, float)):
                        elapsed_ms = max((mount_end - float(start_trigger)) * 1000.0, mount_latency_ms)
                    table_entry["signature"] = table_signature
                    table_entry["rendered"] = True
                    table_entry["updated_at"] = table_timestamp
                    table_entry["dataset_hash"] = dataset_token
                    table_entry["lazy_load_ms"] = elapsed_ms
                    table_entry["mount_latency_ms"] = mount_latency_ms
                    table_refs["dataset_hash"] = dataset_token
                    with _VISUAL_STATE_LOCK:
                        visual_cache_entry["table_placeholder"] = placeholder
                        visual_cache_entry["table_rendered"] = True
                        visual_cache_entry["table_timestamp"] = table_timestamp
                        visual_cache_entry["table_lazy_load_ms"] = elapsed_ms
                        visual_cache_entry["table_mount_latency_ms"] = mount_latency_ms
                    _store_component_metadata(
                        portfolio_id,
                        table_filters,
                        tab_slug,
                        "table",
                        table_timestamp,
                    )
                    completed_at = table_timestamp
                    table_lazy["triggered_at"] = None
                    table_lazy["loaded_at"] = completed_at
                    table_lazy["lazy_load_ms"] = elapsed_ms
                    table_lazy["mount_latency_ms"] = mount_latency_ms
                    if mount_latency_ms > 100.0:
                        logger.warning(
                            "La visualización 'table' tardó %.1f ms en montarse (dataset=%s)",
                            mount_latency_ms,
                            dataset_token,
                        )
                    _record_lazy_component_load(
                        "table",
                        elapsed_ms,
                        dataset_token,
                        mount_latency_ms=mount_latency_ms,
                    )
                elif table_meta is not None:
                    table_entry.setdefault("updated_at", table_meta.get("computed_at"))
                    table_entry.setdefault("signature", table_signature)
                    table_refs.setdefault("dataset_hash", dataset_token)
                    with _VISUAL_STATE_LOCK:
                        visual_cache_entry.setdefault("table_placeholder", table_placeholder)
                        visual_cache_entry.setdefault("table_rendered", True)
                        visual_cache_entry.setdefault(
                            "table_lazy_load_ms", table_entry.get("lazy_load_ms")
                        )
                        visual_cache_entry.setdefault(
                            "table_mount_latency_ms", table_entry.get("mount_latency_ms")
                        )
                        visual_cache_entry.setdefault(
                            "table_timestamp", table_entry.get("updated_at")
                        )

        if table_ready:
            table_ctx.stop()

    if table_ready or table_entry.get("rendered"):
        table_reused = (
            bool(table_entry.get("rendered"))
            and table_entry.get("dataset_hash") == dataset_token
            and table_entry.get("signature") == table_signature
        )
        visual_cache_registry.record(
            "table",
            dataset_hash=dataset_hash,
            reused=table_reused,
            signature=table_signature,
        )

    charts_signature = (portfolio_id, chart_filters)
    charts_meta = _get_component_metadata(portfolio_id, chart_filters, tab_slug, "charts")
    charts_entry_hash = charts_entry.get("dataset_hash")
    charts_entry_dataset = charts_entry.get("dataset_hash")
    charts_entry.setdefault("dataset_hash", charts_entry_hash or dataset_token)
    charts_placeholder = charts_entry.get("body_placeholder") or charts_entry["placeholder"]
    charts_trigger_placeholder = charts_entry.get("trigger_placeholder") or charts_entry["placeholder"]
    charts_refs["placeholder"] = charts_placeholder
    previously_rendered_charts = bool(charts_entry.get("rendered"))
    if _should_reset_rendered_flag(
        charts_entry_dataset,
        dataset_token,
        charts_lazy.get("status"),
        soft_refresh_guard=soft_refresh_guard,
    ):
        charts_entry["rendered"] = False
        previously_rendered_charts = False

    with charts_fragment(dataset_token=dataset_token) as charts_ctx:
        charts_ready = _prompt_lazy_block(
            charts_lazy,
            placeholder=charts_trigger_placeholder,
            button_label="📈 Cargar gráficos del portafolio",
            info_message="Los gráficos intradía y el heatmap se cargarán bajo demanda.",
            key=f"{tab_slug}_load_charts",
            dataset_token=dataset_token,
            fallback_key="load_charts",
        )

        should_render_charts = False
        if charts_ready:
            with _record_stage("render_charts", timings):
                should_render_charts = (
                    charts_entry.get("dataset_hash") != dataset_token
                    or not charts_entry.get("rendered")
                )
                if should_render_charts:
                    placeholder = charts_placeholder
                    if not previously_rendered_charts:
                        skeletons.mark_placeholder("charts", placeholder=placeholder)
                        try:
                            placeholder.write("⏳ Cargando gráficos del portafolio…")
                        except Exception:  # pragma: no cover - defensive guard for Streamlit stubs
                            logger.debug(
                                "No se pudo mostrar el placeholder de carga de gráficos",
                                exc_info=True,
                            )
                    references = charts_refs.get("references")
                    start_trigger = charts_lazy.get("triggered_at")
                    spinner_factory = getattr(st, "spinner", None)
                    spinner_cm = (
                        spinner_factory("Procesando gráficos del portafolio…")
                        if callable(spinner_factory)
                        else nullcontext()
                    )
                    mount_start = time.perf_counter()
                    with spinner_cm:
                        updated_refs = update_charts(
                            placeholder,
                            render_fn=render_charts_section,
                            df_view=df_view,
                            controls=controls,
                            ccl_rate=ccl_rate,
                            totals=totals,
                            contribution_metrics=contributions,
                            snapshot=snapshot,
                            references=references,
                        )
                    mount_end = time.perf_counter()
                    charts_refs["references"] = updated_refs
                    charts_timestamp = time.time()
                    mount_latency_ms = max((mount_end - mount_start) * 1000.0, 0.0)
                    elapsed_ms = mount_latency_ms
                    if isinstance(start_trigger, (int, float)):
                        elapsed_ms = max((mount_end - float(start_trigger)) * 1000.0, mount_latency_ms)
                    charts_entry["signature"] = charts_signature
                    charts_entry["rendered"] = True
                    charts_entry["updated_at"] = charts_timestamp
                    charts_entry["dataset_hash"] = dataset_token
                    charts_entry["lazy_load_ms"] = elapsed_ms
                    charts_entry["mount_latency_ms"] = mount_latency_ms
                    charts_refs["dataset_hash"] = dataset_token
                    with _VISUAL_STATE_LOCK:
                        visual_cache_entry["charts_placeholder"] = placeholder
                        visual_cache_entry["charts_rendered"] = True
                        visual_cache_entry["charts_timestamp"] = charts_timestamp
                        visual_cache_entry["charts_lazy_load_ms"] = elapsed_ms
                        visual_cache_entry["charts_mount_latency_ms"] = mount_latency_ms
                    _store_component_metadata(
                        portfolio_id,
                        chart_filters,
                        tab_slug,
                        "charts",
                        charts_timestamp,
                    )
                    completed_at = charts_timestamp
                    charts_lazy["triggered_at"] = None
                    charts_lazy["loaded_at"] = completed_at
                    charts_lazy["lazy_load_ms"] = elapsed_ms
                    charts_lazy["mount_latency_ms"] = mount_latency_ms
                    if mount_latency_ms > 100.0:
                        logger.warning(
                            "La visualización 'charts' tardó %.1f ms en montarse (dataset=%s)",
                            mount_latency_ms,
                            dataset_token,
                        )
                    _record_lazy_component_load(
                        "chart",
                        elapsed_ms,
                        dataset_token,
                        mount_latency_ms=mount_latency_ms,
                    )
                elif charts_meta is not None:
                    charts_entry.setdefault("updated_at", charts_meta.get("computed_at"))
                    charts_entry.setdefault("signature", charts_signature)
                    charts_refs.setdefault("dataset_hash", dataset_token)
                    with _VISUAL_STATE_LOCK:
                        visual_cache_entry.setdefault(
                            "charts_placeholder", charts_entry.get("placeholder")
                        )
                        visual_cache_entry.setdefault("charts_rendered", True)
                        visual_cache_entry.setdefault(
                            "charts_lazy_load_ms", charts_entry.get("lazy_load_ms")
                        )
                        visual_cache_entry.setdefault(
                            "charts_mount_latency_ms", charts_entry.get("mount_latency_ms")
                        )
                        visual_cache_entry.setdefault(
                            "charts_timestamp", charts_entry.get("updated_at")
                        )

        if charts_ready:
            charts_ctx.stop()

    if charts_ready or charts_entry.get("rendered"):
        charts_reused = (
            bool(charts_entry.get("rendered"))
            and charts_entry.get("dataset_hash") == dataset_token
            and charts_entry.get("signature") == charts_signature
        )
        visual_cache_registry.record(
            "charts",
            dataset_hash=dataset_hash,
            reused=charts_reused,
            signature=charts_signature,
        )

    partial_update_ms = (time.perf_counter() - partial_update_start) * 1000.0
    incremental_render = not should_render_summary and not should_render_table and not should_render_charts
    if isinstance(tab_cache, dict):
        tab_cache["lazy_pending"] = (
            table_lazy.get("status") != "loaded" or charts_lazy.get("status") != "loaded"
        )
    if not incremental_render:
        cached_dataset = render_refs.get("dataset_hash")
        if cached_dataset == dataset_token:
            incremental_render = (
                previously_rendered_summary
                and previously_rendered_table
                and previously_rendered_charts
                and not should_render_summary
                and not should_render_table
                and not should_render_charts
            )
    render_refs["dataset_hash"] = dataset_token
    render_refs["incremental_render"] = incremental_render
    render_refs["ui_partial_update_ms"] = partial_update_ms
    with _VISUAL_STATE_LOCK:
        visual_cache_entry["incremental_render"] = incremental_render
        visual_cache_entry["ui_partial_update_ms"] = partial_update_ms
    try:
        st.session_state["portfolio_incremental_render"] = incremental_render
        st.session_state["portfolio_ui_partial_update_ms"] = partial_update_ms
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug(
            "No se pudo persistir el estado incremental del portafolio", exc_info=True
        )


def render_risk_tab(
    df_view,
    tasvc: TAService,
    favorites,
    notifications: NotificationFlags,
    *,
    available_types: Sequence[str] | None = None,
) -> None:
    """Render risk analysis information for the given snapshot."""

    risk_module = _load_risk_module()
    if risk_module is None:
        logger.debug("Risk module not available; skipping risk tab rendering")
        return

    risk_module.render_risk_analysis(
        df_view,
        tasvc,
        favorites=favorites,
        notifications=notifications,
        available_types=available_types,
    )


def render_fundamentals_tab(
    df_view,
    tasvc: TAService,
    favorites,
    notifications: NotificationFlags,
) -> None:
    """Render fundamentals tab using the given data sources."""

    render_fundamental_analysis(
        df_view,
        tasvc,
        favorites=favorites,
        notifications=notifications,
    )


def render_notifications_panel(
    favorites,
    notifications: NotificationFlags,
    *,
    ui: Any = st,
) -> None:
    """Render badges and indicators for the notifications panel."""

    if notifications.technical_signal:
        render_technical_badge(
            help_text="Tenés señales técnicas recientes para revisar en tus activos favoritos.",
        )
    render_favorite_badges(
        favorites,
        empty_message="⭐ Aún no marcaste favoritos para seguimiento rápido.",
    )


def _select_first(options: Iterable[str]) -> str | None:
    for item in options:
        return item
    return None


def render_technical_tab(
    tasvc: TAService,
    favorites,
    notifications: NotificationFlags,
    all_symbols: list[str],
    viewmodel,
    *,
    map_symbol: Callable[[str], str] | None = None,
    ui: Any | None = None,
    timer: Callable[[], float] = time.perf_counter,
    record_latency: Callable[[str, float | None, str], None] = record_tab_latency,
    plot_chart: Callable[..., Any] | None = None,
    render_fundamentals: Callable[[Mapping[str, Any]], None] | None = None,
) -> None:
    """Render the technical indicators tab for a specific symbol selection."""

    if map_symbol is None:
        map_symbol = map_to_us_ticker
    if ui is None:
        ui = st
    if plot_chart is None:
        plot_chart = plot_technical_analysis_chart
    if render_fundamentals is None:
        render_fundamentals = render_fundamental_data

    ui.subheader("Indicadores técnicos por activo")
    render_notifications_panel(favorites, notifications, ui=ui)
    if not all_symbols:
        ui.info("No hay símbolos en el portafolio para analizar.")
        return
    all_symbols_vm = list(viewmodel.metrics.all_symbols)
    if not all_symbols_vm:
        ui.info("No hay símbolos en el portafolio para analizar.")
        return

    options = favorites.sort_options(all_symbols_vm)
    if not options:
        options = all_symbols_vm
    sym = ui.selectbox(
        "Seleccioná un símbolo (CEDEAR / ETF)",
        options=options,
        index=favorites.default_index(options),
        key="ta_symbol",
        format_func=favorites.format_symbol,
    )
    if not sym:
        sym = _select_first(options)
    if not sym:
        ui.info("No hay símbolos en el portafolio para analizar.")
        return

    render_favorite_toggle(
        sym,
        favorites,
        key_prefix="ta",
        help_text="Los favoritos quedan disponibles en todas las secciones.",
    )

    try:
        us_ticker = map_symbol(sym)
    except ValueError:
        ui.info("No se encontró ticker US para este activo.")
        return

    try:
        fundamental_data = tasvc.fundamentals(us_ticker) or {}
    except AppError as err:
        ui.error(str(err))
    except Exception:
        logger.exception("Error al obtener datos fundamentales para %s", sym)
        ui.error("No se pudieron obtener datos fundamentales, intente más tarde")
    else:
        render_fundamentals(fundamental_data)

    cols = ui.columns([1, 1, 1, 1])
    with cols[0]:
        period = ui.selectbox("Período", ["3mo", "6mo", "1y", "2y"], index=1)
    with cols[1]:
        interval = ui.selectbox("Intervalo", ["1d", "1h", "30m"], index=0)
    with cols[2]:
        sma_fast = ui.number_input(
            "SMA corta",
            min_value=5,
            max_value=100,
            value=20,
            step=1,
        )
    with cols[3]:
        sma_slow = ui.number_input(
            "SMA larga",
            min_value=10,
            max_value=250,
            value=50,
            step=5,
        )

    with ui.expander("Parámetros adicionales"):
        c1, c2, c3 = ui.columns(3)
        macd_fast = c1.number_input(
            "MACD rápida", min_value=5, max_value=50, value=12, step=1
        )
        macd_slow = c2.number_input(
            "MACD lenta", min_value=10, max_value=200, value=26, step=1
        )
        macd_signal = c3.number_input(
            "MACD señal", min_value=5, max_value=50, value=9, step=1
        )
        c4, c5, c6 = ui.columns(3)
        atr_win = c4.number_input(
            "ATR ventana", min_value=5, max_value=200, value=14, step=1
        )
        stoch_win = c5.number_input(
            "Estocástico ventana", min_value=5, max_value=200, value=14, step=1
        )
        stoch_smooth = c6.number_input(
            "Estocástico suavizado", min_value=1, max_value=50, value=3, step=1
        )
        c7, c8, c9 = ui.columns(3)
        ichi_conv = c7.number_input(
            "Ichimoku conv.", min_value=1, max_value=50, value=9, step=1
        )
        ichi_base = c8.number_input(
            "Ichimoku base", min_value=2, max_value=100, value=26, step=1
        )
        ichi_span = c9.number_input(
            "Ichimoku span B", min_value=2, max_value=200, value=52, step=1
        )

    indicator_latency: float | None = None
    try:
        start_time = timer()
        df_ind = tasvc.indicators_for(
            sym,
            period=period,
            interval=interval,
            sma_fast=sma_fast,
            sma_slow=sma_slow,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            atr_win=atr_win,
            stoch_win=stoch_win,
            stoch_smooth=stoch_smooth,
            ichi_conv=ichi_conv,
            ichi_base=ichi_base,
            ichi_span=ichi_span,
        )
        indicator_latency = (timer() - start_time) * 1000.0
    except AppError as err:
        if indicator_latency is None:
            indicator_latency = (timer() - start_time) * 1000.0
        record_latency("tecnico", indicator_latency, status="error")
        ui.error(str(err))
        return
    except Exception:
        logger.exception("Error al obtener indicadores técnicos para %s", sym)
        if indicator_latency is None:
            indicator_latency = (timer() - start_time) * 1000.0
        record_latency("tecnico", indicator_latency, status="error")
        ui.error("No se pudieron obtener indicadores técnicos, intente más tarde")
        return
    record_latency("tecnico", indicator_latency, status="success")
    if df_ind.empty:
        ui.info("No se pudo descargar histórico para ese símbolo/periodo/intervalo.")
    else:
        fig = plot_chart(df_ind, sma_fast, sma_slow)
        ui.plotly_chart(
            fig,
            width="stretch",
            key="ta_chart",
            config=PLOTLY_CONFIG,
        )
        ui.caption(
            "Gráfico de precio con indicadores técnicos como "
            "medias móviles, RSI o MACD para detectar tendencias "
            "y señales."
        )
        alerts = tasvc.alerts_for(df_ind)
        if alerts:
            for a in alerts:
                al = a.lower()
                if "bajista" in al or "sobrecompra" in al:
                    ui.warning(a)
                elif "alcista" in al or "sobreventa" in al:
                    ui.success(a)
                else:
                    ui.info(a)
        else:
            ui.caption("Sin alertas técnicas en la última vela.")

        ui.subheader("Backtesting")
        strat = ui.selectbox(
            "Estrategia", ["SMA", "MACD", "Estocástico", "Ichimoku"], index=0
        )
        backtest_latency: float | None = None
        try:
            start_time = timer()
            bt = tasvc.backtest(df_ind, strategy=strat)
            backtest_latency = (timer() - start_time) * 1000.0
        except AppError as err:
            if backtest_latency is None:
                backtest_latency = (timer() - start_time) * 1000.0
            record_latency("tecnico", backtest_latency, status="error")
            ui.error(str(err))
            return
        except Exception:
            logger.exception("Error al ejecutar backtesting para %s", sym)
            if backtest_latency is None:
                backtest_latency = (timer() - start_time) * 1000.0
            record_latency("tecnico", backtest_latency, status="error")
            ui.error("No se pudo ejecutar el backtesting, intente más tarde")
            return
        record_latency("tecnico", backtest_latency, status="success")
        if bt.empty:
            ui.info("Sin datos suficientes para el backtesting.")
        else:
            ui.line_chart(bt["equity"])
            ui.caption(
                "La línea muestra cómo habría crecido la inversión usando la estrategia seleccionada."
            )
            ui.metric("Retorno acumulado", f"{bt['equity'].iloc[-1] - 1:.2%}")


def render_portfolio_section(
    container,
    cli,
    fx_rates,
    *,
    view_model_service_factory: Callable[[], PortfolioViewModelService] | None = None,
    notifications_service_factory: Callable[[], NotificationsService] | None = None,
    timings: dict[str, float] | None = None,
    lazy_metrics: bool = False,
) -> Any:
    """Render the main portfolio section and return refresh interval."""
    with container:
        psvc = get_portfolio_service()
        tasvc = get_ta_service()

        view_model_service = get_portfolio_view_service(view_model_service_factory)
        notifications_service = get_notifications_service(notifications_service_factory)

        visual_cache_cleared = _maybe_reset_visual_cache_state()

        def _schedule_lazy_metrics_refresh() -> None:
            thread_key = "portfolio_extended_thread"
            dataset_key = view_model_service._hash_dataset(df_pos)
            entry = st.session_state.get(thread_key)
            if isinstance(entry, Mapping):
                active_thread = entry.get("thread")
                active_dataset = entry.get("dataset")
                if (
                    isinstance(active_thread, threading.Thread)
                    and active_thread.is_alive()
                    and active_dataset == dataset_key
                ):
                    return

            def _compute_and_rerun() -> None:
                try:
                    current_dataset = getattr(view_model_service, "_dataset_key", None)
                    if current_dataset != dataset_key:
                        return
                    view_model_service.compute_extended_metrics(
                        df_pos=df_pos,
                        controls=controls,
                        cli=cli,
                        psvc=psvc,
                    )
                except Exception:  # pragma: no cover - defensive safeguard
                    logger.debug(
                        "No se pudo calcular métricas extendidas del portafolio",
                        exc_info=True,
                    )
                finally:
                    try:
                        st.session_state.pop(thread_key, None)
                    except Exception:
                        logger.debug(
                            "No se pudo limpiar el estado del hilo de métricas extendidas",
                            exc_info=True,
                        )
                    try:
                        st.experimental_rerun()
                    except Exception:
                        logger.debug(
                            "No se pudo solicitar rerun tras completar métricas extendidas",
                            exc_info=True,
                        )

            worker = threading.Thread(
                target=_compute_and_rerun,
                name="portfolio-extended-metrics",
                daemon=True,
            )

            try:  # pragma: no cover - optional context for Streamlit threads
                from streamlit.runtime.scriptrunner import (
                    add_script_run_ctx,
                    get_script_run_ctx,
                )
            except Exception:
                add_script_run_ctx = get_script_run_ctx = None
            if add_script_run_ctx and get_script_run_ctx:
                try:
                    add_script_run_ctx(worker, ctx=get_script_run_ctx())
                except Exception:
                    logger.debug(
                        "No se pudo adjuntar el contexto de Streamlit al hilo de métricas",
                        exc_info=True,
                    )

            worker.start()
            st.session_state[thread_key] = {"thread": worker, "dataset": dataset_key}

        if snapshot_service.is_null_backend():
            backend_name = snapshot_service.current_backend_name()
            st.warning(
                "El almacenamiento de snapshots está deshabilitado "
                f"(backend: {backend_name}). Configurá `SNAPSHOT_BACKEND` a "
                "`json` o `sqlite` en `config.json` (o llamando a "
                "`services.snapshots.configure_storage`) y verificá los permisos "
                "definidos en `SNAPSHOT_STORAGE_PATH` para volver a habilitarlo."
            )

        with _record_stage("load_data", timings):
            df_pos, all_symbols, available_types = load_portfolio_data(cli, psvc)

        favorites = _get_cached_favorites()

        with _record_stage("apply_filters", timings):
            controls: Controls = render_sidebar(
                all_symbols,
                available_types,
            )

        controls_snapshot = asdict(controls)
        controls_changed = False
        try:
            previous_controls = st.session_state.get(_CONTROLS_SNAPSHOT_STATE_KEY)
        except Exception:  # pragma: no cover - defensive safeguard
            previous_controls = None
        if not isinstance(previous_controls, dict) or previous_controls != controls_snapshot:
            controls_changed = True
        try:
            st.session_state[_CONTROLS_SNAPSHOT_STATE_KEY] = dict(controls_snapshot)
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug("No se pudo persistir el snapshot de controles", exc_info=True)

        refresh_secs = controls.refresh_secs
        precomputed_dataset_hash: str | None = None
        hash_helper = getattr(view_model_service, "_hash_dataset", None)
        if callable(hash_helper):
            try:
                precomputed_dataset_hash = str(hash_helper(df_pos))
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug(
                    "No se pudo calcular el hash del dataset previo al viewmodel",
                    exc_info=True,
                )
                precomputed_dataset_hash = None
        skip_invalidation_flag = bool(
            st.session_state.pop("_dataset_skip_invalidation", False)
        )

        if skip_invalidation_flag:
            logger.info(
                "[Guardian] Prevented pre-render cache invalidation",
                extra={"dataset_hash": precomputed_dataset_hash},
            )

        soft_refresh_guard = False

        with _record_stage("build_viewmodel", timings):
            snapshot = view_model_service.get_portfolio_view(
                df_pos=df_pos,
                controls=controls,
                cli=cli,
                psvc=psvc,
                lazy_metrics=lazy_metrics,
                dataset_hash=precomputed_dataset_hash,
                skip_invalidation=skip_invalidation_flag,
            )

            soft_refresh_guard = bool(
                getattr(snapshot, "soft_refresh_guard", False)
            )

            viewmodel = build_portfolio_viewmodel(
                snapshot=snapshot,
                controls=controls,
                fx_rates=fx_rates,
                all_symbols=all_symbols,
            )

        try:
            st.session_state["portfolio_last_viewmodel"] = viewmodel
            st.session_state["portfolio_last_positions"] = viewmodel.positions
            st.session_state["portfolio_last_totals"] = viewmodel.totals
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug("No se pudo almacenar el viewmodel en session_state", exc_info=True)

        pending_extended = "extended_metrics" in getattr(viewmodel, "pending_metrics", ())
        if lazy_metrics and pending_extended:
            _schedule_lazy_metrics_refresh()

        with _record_stage("notifications", timings):
            notifications = notifications_service.get_flags()
        tab_labels = _apply_tab_badges(list(viewmodel.tab_options), notifications)

        tab_idx = st.radio(
            "Secciones",
            options=range(len(tab_labels)),
            format_func=lambda i: tab_labels[i],
            horizontal=True,
            key="portfolio_tab",
        )
        df_view = viewmodel.positions

        dataset_hash = snapshot.dataset_hash or precomputed_dataset_hash
        if not dataset_hash:
            try:
                dataset_hash = _hash_dataframe(df_view)
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug("No se pudo calcular el hash del dataset del portafolio", exc_info=True)
                dataset_hash = ""
        dataset_hash = str(dataset_hash or "")
        if controls_changed:
            log_user_action("filter_change", controls_snapshot, dataset_hash=dataset_hash)
        manual_refresh_flag = bool(st.session_state.pop("refresh_pending", False))
        previous_hash = st.session_state.get(_DATASET_HASH_STATE_KEY)
        if (
            previous_hash
            and dataset_hash
            and previous_hash != dataset_hash
            and not skip_invalidation_flag
        ):
            visual_cache_registry.invalidate_dataset(
                previous_hash, reason="dataset_hash_changed"
            )
            _log_quotes_refresh_event(
                dataset_hash,
                source="dataset_update",
                detail={"previous_hash": previous_hash},
            )
        reused_visual_cache = bool(previous_hash) and previous_hash == dataset_hash
        try:
            st.session_state[_DATASET_HASH_STATE_KEY] = dataset_hash
            st.session_state[_DATASET_REUSE_FLAG_KEY] = reused_visual_cache
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug("No se pudo actualizar el estado de caché visual", exc_info=True)
        visual_entry = _get_visual_cache_entry(dataset_hash)
        dataset_token = str(dataset_hash or "none")
        with _VISUAL_STATE_LOCK:
            visual_entry["dataset_hash"] = dataset_hash
            visual_entry["last_seen"] = time.time()

        if soft_refresh_guard:
            if dataset_hash:
                logger.info(
                    "[Guardian] Soft refresh intercepted before UI reset",
                    extra={"dataset_hash": dataset_hash},
                )
            fragment_state_soft_refresh(dataset_hash=dataset_hash)
            try:
                st.session_state["_soft_refresh_applied"] = True
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug(
                    "No se pudo registrar el soft refresh activo", exc_info=True
                )
        else:
            try:
                st.session_state.pop("_soft_refresh_applied", None)
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug(
                    "No se pudo limpiar la marca de soft refresh", exc_info=True
                )

        if manual_refresh_flag:
            last_refresh = st.session_state.get(_REFRESH_LOG_STATE_KEY)
            if last_refresh != dataset_hash:
                _log_quotes_refresh_event(dataset_hash, source="manual_refresh")
                try:
                    st.session_state[_REFRESH_LOG_STATE_KEY] = dataset_hash
                except Exception:  # pragma: no cover - defensive safeguard
                    logger.debug("No se pudo registrar el refresh manual", exc_info=True)

        try:
            base_label = viewmodel.tab_options[tab_idx]
        except (IndexError, TypeError):
            base_label = str(tab_idx)
        tab_slug = _slugify_metric_label(base_label)

        _set_active_tab(tab_slug)
        if isinstance(df_view, pd.DataFrame) and df_view.empty:
            try:
                last_empty = st.session_state.get(_EMPTY_RENDER_LOG_KEY)
            except Exception:  # pragma: no cover - defensive safeguard
                last_empty = None
            if last_empty != dataset_hash:
                log_user_action(
                    "empty_dataframe_render",
                    {"tab": tab_slug},
                    dataset_hash=dataset_hash,
                )
                try:
                    st.session_state[_EMPTY_RENDER_LOG_KEY] = dataset_hash
                except Exception:  # pragma: no cover - defensive safeguard
                    logger.debug("No se pudo registrar el render vacío", exc_info=True)
        render_cache = _ensure_render_cache()
        cache_entry = _ensure_tab_cache(render_cache, tab_slug)
        tab_signature = _tab_signature(viewmodel, df_view, tab_slug)
        tab_loaded = st.session_state.get("tab_loaded")
        if not isinstance(tab_loaded, dict):
            tab_loaded = {}
            st.session_state["tab_loaded"] = tab_loaded

        with _record_stage(f"render_tab.{tab_slug}", timings):
            first_visit = not bool(tab_loaded.get(tab_slug))
            lazy_pending = bool(cache_entry.get("lazy_pending"))
            should_render = (
                cache_entry.get("signature") != tab_signature
                or not cache_entry.get("rendered")
                or lazy_pending
            )
            source = "fresh" if not cache_entry.get("rendered") else "cache"
            latency_ms: float | None = cache_entry.get("latency_ms")
            rendered = False
            if should_render:
                if cache_entry.get("rendered"):
                    source = "hot"
                body_placeholder = cache_entry["body_placeholder"]
                perf_start = time.perf_counter()
                with body_placeholder.container():
                    if first_visit:
                        spinner_cm = st.spinner("Cargando pestaña...")
                    else:
                        spinner_cm = nullcontext()
                    with spinner_cm:
                        with profile_block(f"render_tab.{tab_slug}"):
                            _render_selected_tab(
                                tab_idx,
                                df_view,
                                tasvc,
                                favorites,
                                notifications,
                                available_types,
                                all_symbols,
                                viewmodel,
                                snapshot,
                                tab_slug=tab_slug,
                                tab_cache=cache_entry,
                                timings=timings,
                                lazy_metrics=lazy_metrics,
                            )
                latency_ms = (time.perf_counter() - perf_start) * 1000.0
                rendered = True
                cache_entry["signature"] = tab_signature
                cache_entry["rendered"] = True
                cache_entry["latency_ms"] = latency_ms
                record_tab_latency(tab_slug, latency_ms, status=source)
            else:
                record_tab_latency(tab_slug, 0.0, status="cache")
                # When nothing was re-rendered we are effectively performing an
                # incremental update and should expose that state for
                # downstream telemetry (even if the render helpers were not
                # invoked on this pass).
                try:
                    st.session_state["portfolio_incremental_render"] = True
                    st.session_state["portfolio_ui_partial_update_ms"] = 0.0
                except Exception:  # pragma: no cover - defensive safeguard
                    logger.debug(
                        "No se pudo marcar el render incremental del portafolio",
                        exc_info=True,
                    )
                render_refs = st.session_state.get(_RENDER_REFS_STATE_KEY)
                if isinstance(render_refs, dict):
                    render_refs["dataset_hash"] = dataset_token
                    render_refs["incremental_render"] = True
                    render_refs["ui_partial_update_ms"] = 0.0
                with _VISUAL_STATE_LOCK:
                    visual_entry["incremental_render"] = True
                    visual_entry["ui_partial_update_ms"] = 0.0

            cache_entry["last_source"] = source
            _update_status_message(
                cache_entry["info_placeholder"],
                base_label,
                latency_ms,
                source,
            )
            tab_loaded[tab_slug] = True

        profile_ms: float | None = None
        overhead_ms: float | None = None
        if latency_ms is not None and isinstance(timings, dict):
            stage_key = f"render_tab.{tab_slug}"
            stage_value = timings.get(stage_key)
            try:
                profile_ms = float(stage_value) if stage_value is not None else None
            except (TypeError, ValueError):
                profile_ms = None
        if profile_ms is not None and latency_ms is not None:
            overhead_ms = max(latency_ms - profile_ms, 0.0)
        elif latency_ms is not None:
            overhead_ms = max(latency_ms, 0.0)

        incremental_render = bool(
            st.session_state.get("portfolio_incremental_render", False)
        )
        partial_update_ms = st.session_state.get("portfolio_ui_partial_update_ms")
        if not isinstance(partial_update_ms, (int, float)):
            partial_update_ms = None
        elif partial_update_ms < 0:
            partial_update_ms = 0.0
        else:
            partial_update_ms = round(float(partial_update_ms), 2)
        if rendered and latency_ms is not None:
            _append_tab_metric(
                tab_slug,
                latency_ms / 1000.0,
                profile_ms=profile_ms,
                overhead_ms=overhead_ms,
            )
        elif incremental_render:
            _append_tab_metric(
                tab_slug,
                0.0,
                profile_ms=0.0,
                overhead_ms=0.0,
            )

        try:
            dataset_hash = st.session_state.get(_DATASET_HASH_STATE_KEY)
        except Exception:  # pragma: no cover - defensive safeguard
            dataset_hash = None
        reused_visual_cache = bool(
            st.session_state.get(_DATASET_REUSE_FLAG_KEY)
        )
        streamlit_overhead = overhead_ms
        if streamlit_overhead is None and reused_visual_cache:
            streamlit_overhead = 0.0
        try:
            ui_persist_ms = st.session_state.get(_UI_PERSIST_STATE_KEY)
        except Exception:  # pragma: no cover - defensive safeguard
            ui_persist_ms = None
        if isinstance(ui_persist_ms, (int, float)):
            ui_persist_ms = round(float(ui_persist_ms), 2)
        else:
            ui_persist_ms = None

        dataset_cache_key = str(dataset_hash or "none")
        try:
            cache_event_state = st.session_state.get(_VISUAL_CACHE_EVENT_STATE_KEY)
        except Exception:  # pragma: no cover - defensive safeguard
            cache_event_state = None
        if not isinstance(cache_event_state, dict):
            cache_event_state = {}
        if visual_cache_cleared:
            cache_event_state[dataset_cache_key] = True
        elif cache_event_state.get(dataset_cache_key):
            visual_cache_cleared = True
        else:
            cache_event_state[dataset_cache_key] = False
        try:
            st.session_state[_VISUAL_CACHE_EVENT_STATE_KEY] = cache_event_state
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug("No se pudo persistir el estado de eventos de caché visual", exc_info=True)

        telemetry_extra = {
            "reused_visual_cache": reused_visual_cache,
            "streamlit_overhead_ms": streamlit_overhead if streamlit_overhead is not None else "",
            "visual_cache_cleared": visual_cache_cleared,
            "incremental_render": incremental_render,
            "ui_partial_update_ms": partial_update_ms,
            "ui_persist_ms": ui_persist_ms if ui_persist_ms is not None else "",
        }
        try:
            log_telemetry(
                (Path("performance_metrics_15.csv"),),
                phase="portfolio.visual_cache",
                dataset_hash=(str(dataset_hash) if dataset_hash else None),
                extra=telemetry_extra,
            )
        except Exception:  # pragma: no cover - telemetry should not break rendering
            logger.debug(
                "No se pudo registrar telemetría de caché visual del portafolio",
                exc_info=True,
            )

        _publish_visual_cache_debug()

        return refresh_secs
@contextmanager
def _record_stage(name: str, timings: dict[str, float] | None = None) -> Iterator[None]:
    """Measure a render stage while recording diagnostics and timings."""

    start = time.perf_counter()
    metric_name = f"portfolio_ui.{name}"
    with measure_execution(metric_name):
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            if timings is not None:
                timings[name] = round(elapsed, 2)


def _slugify_metric_label(label: str) -> str:
    """Return a slug suitable for metric identifiers based on ``label``."""

    normalized = unicodedata.normalize("NFKD", str(label))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized or "tab"


@st.cache_resource(show_spinner=False)
def _get_cached_favorites() -> FavoriteSymbols:
    """Return the persistent favorites manager using Streamlit's resource cache."""

    return get_persistent_favorites()


def _set_active_tab(tab_slug: str) -> None:
    """Persist the currently active tab slug in the session state."""

    try:
        previous = st.session_state.get("active_tab")
    except Exception:  # pragma: no cover - defensive safeguard
        previous = None
    if previous != tab_slug:
        detail = {"from": previous or "", "to": tab_slug}
        log_user_action("tab_change", detail, dataset_hash=_current_dataset_hash())
    try:
        st.session_state["active_tab"] = tab_slug
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo actualizar active_tab en session_state", exc_info=True)


def _ensure_render_cache() -> dict[str, dict[str, Any]]:
    """Return the mutable render cache stored in the session state."""

    cache = st.session_state.get("render_cache")
    if not isinstance(cache, dict):
        cache = {}
        st.session_state["render_cache"] = cache
    return cache


def _ensure_tab_cache(cache: dict[str, dict[str, Any]], tab_slug: str) -> dict[str, Any]:
    """Ensure cache scaffolding for ``tab_slug`` and return the entry."""

    entry: dict[str, Any]
    raw_entry = cache.get(tab_slug)
    if isinstance(raw_entry, dict):
        entry = raw_entry
    else:
        entry = {}
        cache[tab_slug] = entry

    info_placeholder = entry.get("info_placeholder")
    if not hasattr(info_placeholder, "markdown"):
        info_placeholder = st.empty()
        entry["info_placeholder"] = info_placeholder

    body_placeholder = entry.get("body_placeholder")
    if not hasattr(body_placeholder, "container"):
        body_placeholder = st.empty()
        entry["body_placeholder"] = body_placeholder

    return entry


def _ensure_visual_cache() -> dict[str, dict[str, Any]]:
    """Return (and initialize) the dataset-level visual cache."""

    with _VISUAL_STATE_LOCK:
        cache = st.session_state.get(_VISUAL_CACHE_STATE_KEY)
        if not isinstance(cache, dict):
            cache = {}
            try:
                st.session_state[_VISUAL_CACHE_STATE_KEY] = cache
            except Exception:  # pragma: no cover - defensive safeguard
                pass
        return cache


def _get_visual_cache_entry(dataset_hash: str | None) -> dict[str, Any]:
    """Return the cache bucket associated with ``dataset_hash``."""

    with _VISUAL_STATE_LOCK:
        cache = _ensure_visual_cache()
        key = str(dataset_hash or "none")
        entry = cache.get(key)
        if not isinstance(entry, dict):
            entry = {}
            cache[key] = entry
        return entry


def _normalize_user_identifier(value: Any) -> str | None:
    """Normalize a user identifier from ``value`` if possible."""

    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, (int, float)):
        try:
            return str(int(value)) if isinstance(value, int) else str(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive branch
            return None
    return None


def _resolve_active_user_id() -> str | None:
    """Return the active user identifier available in the session state."""

    user_id = get_current_user_id()
    if user_id:
        return user_id
    try:
        raw = st.session_state.get("last_user_id")
    except Exception:  # pragma: no cover - defensive safeguard
        return None
    return _normalize_user_identifier(raw)


def _clear_visual_cache_state() -> None:
    """Remove cached visual state from ``st.session_state``."""

    visual_cache_registry.invalidate_all(reason="session_state_reset")
    with _VISUAL_STATE_LOCK:
        for key in (
            _VISUAL_CACHE_STATE_KEY,
            _DATASET_HASH_STATE_KEY,
            _RENDER_REFS_STATE_KEY,
            _LAZY_BLOCKS_STATE_KEY,
            _VISUAL_CACHE_EVENT_STATE_KEY,
            _UI_PERSIST_STATE_KEY,
            "portfolio_incremental_render",
            "portfolio_ui_partial_update_ms",
        ):
            try:
                if key in st.session_state:
                    del st.session_state[key]
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug("No se pudo limpiar el estado %s", key, exc_info=True)
        try:
            st.session_state.pop(_DATASET_REUSE_FLAG_KEY, None)
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug(
                "No se pudo limpiar la bandera de reutilización de caché visual",
                exc_info=True,
            )


def _maybe_reset_visual_cache_state() -> bool:
    """Reset visual caches when the active user changes or logs out."""

    state = getattr(st, "session_state", None)
    if state is None:
        return False

    current_user = _resolve_active_user_id()
    previous_user = _normalize_user_identifier(state.get(_PORTFOLIO_LAST_USER_STATE_KEY))
    should_reset = False
    if current_user is None:
        if previous_user is not None:
            should_reset = True
    elif previous_user is not None and previous_user != current_user:
        should_reset = True

    if should_reset:
        _clear_visual_cache_state()
        session_logger.info("Visual cache cleared due to user change/logout")
        try:
            dataset_hash = state.get(_DATASET_HASH_STATE_KEY)
        except Exception:  # pragma: no cover - defensive safeguard
            dataset_hash = None
        detail = {"previous_user": previous_user or "", "current_user": current_user or ""}
        action = "logout" if current_user is None else "user_switch"
        log_user_action(action, detail, dataset_hash=str(dataset_hash or ""))

    try:
        with _VISUAL_STATE_LOCK:
            if current_user is None:
                state.pop(_PORTFOLIO_LAST_USER_STATE_KEY, None)
            else:
                state[_PORTFOLIO_LAST_USER_STATE_KEY] = current_user
                legacy_user = _normalize_user_identifier(state.get("last_user_id"))
                if legacy_user != current_user:
                    state["last_user_id"] = current_user
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug(
            "No se pudo actualizar el estado de usuario del portafolio",
            exc_info=True,
        )

    return should_reset


def _tab_signature(viewmodel: Any, df_view: Any, tab_slug: str) -> tuple[Any, ...]:
    """Build a lightweight signature to detect content changes per tab."""

    snapshot_id = getattr(viewmodel, "snapshot_id", None)
    controls = getattr(viewmodel, "controls", None)
    dataset_key = _portfolio_dataset_key(viewmodel, df_view)
    try:
        order_by = getattr(controls, "order_by", None)
        desc = getattr(controls, "desc", None)
        show_usd = getattr(controls, "show_usd", None)
        top_n = getattr(controls, "top_n", None)
        selected_syms = tuple(sorted(str(sym) for sym in getattr(controls, "selected_syms", []) or ()))
        selected_types = tuple(sorted(str(tp) for tp in getattr(controls, "selected_types", []) or ()))
        symbol_query = (getattr(controls, "symbol_query", "") or "").strip()
        hide_cash = getattr(controls, "hide_cash", None)
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo calcular la firma de la pestaña %s", tab_slug, exc_info=True)
        order_by = desc = show_usd = top_n = symbol_query = hide_cash = None
        selected_syms = ()
        selected_types = ()
    return (
        tab_slug,
        snapshot_id,
        dataset_key,
        order_by,
        desc,
        show_usd,
        top_n,
        selected_syms,
        selected_types,
        symbol_query,
        hide_cash,
        tuple(sorted(str(item) for item in getattr(viewmodel, "pending_metrics", ()) or ())),
    )


def _render_selected_tab(
    tab_idx: int,
    df_view,
    tasvc: TAService,
    favorites,
    notifications: NotificationFlags,
    available_types: Sequence[str] | None,
    all_symbols: list[str],
    viewmodel,
    snapshot,
    *,
    tab_slug: str,
    tab_cache: dict[str, Any] | None = None,
    timings: dict[str, float] | None = None,
    lazy_metrics: bool = False,
) -> None:
    """Dispatch rendering to the appropriate tab implementation."""

    if tab_idx == 0:
        render_basic_tab(
            viewmodel,
            favorites,
            snapshot,
            tab_slug=tab_slug,
            tab_cache=tab_cache,
            timings=timings,
            lazy_metrics=lazy_metrics,
        )
    elif tab_idx == 1:
        render_advanced_analysis(df_view, tasvc)
    elif tab_idx == 2:
        render_risk_tab(
            df_view,
            tasvc,
            favorites,
            notifications,
            available_types=available_types,
        )
    elif tab_idx == 3:
        render_fundamentals_tab(df_view, tasvc, favorites, notifications)
    else:
        render_technical_tab(
            tasvc,
            favorites,
            notifications,
            all_symbols,
            viewmodel,
        )


_SOURCE_LABELS = {
    "fresh": "cálculo inicial",
    "hot": "recalculado",
    "cache": "caché en memoria",
}


def _update_status_message(placeholder, base_label: str, latency_ms: float | None, source: str) -> None:
    """Render a status line with latency metadata for the active tab."""

    label = base_label.strip() or "Sección"
    latency_text = "–" if latency_ms is None else f"{latency_ms:.0f} ms"
    source_label = _SOURCE_LABELS.get(source, source)
    message = f"**{label}** · {latency_text} · Fuente: {source_label}"
    try:
        placeholder.markdown(message)
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo actualizar el estado de la pestaña %s", base_label, exc_info=True)
