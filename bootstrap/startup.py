"""Startup helpers for worker orchestration and telemetry."""

from __future__ import annotations

import importlib
import logging
import threading
import time
from collections.abc import Callable, Iterable
from functools import lru_cache
from types import ModuleType

import streamlit as st

from bootstrap.config import TOTAL_LOAD_START
from services.startup_logger import log_startup_event
from shared.settings import (
    enable_prometheus,
    performance_store_ttl_days,
    sqlite_maintenance_interval_hours,
    sqlite_maintenance_size_threshold_mb,
)

logger = logging.getLogger(__name__)


def _resolve_streamlit_singleton():
    if hasattr(st, "singleton"):
        return st.singleton

    if hasattr(st, "cache_resource"):
        logger.debug("Streamlit.singleton no disponible; usando cache_resource como reemplazo")

        def _cache_resource_singleton(*, show_spinner: bool = False):
            return st.cache_resource(show_spinner=show_spinner)

        return _cache_resource_singleton

    if hasattr(st, "experimental_singleton"):
        logger.debug(
            "Streamlit.singleton no disponible; usando experimental_singleton como reemplazo"
        )

        def _experimental_singleton(*, show_spinner: bool = False):
            return st.experimental_singleton(show_spinner=show_spinner)

        return _experimental_singleton

    logger.warning(
        "Streamlit no expone decoradores singleton compatibles; se usará un passthrough"
    )

    def _passthrough(*, show_spinner: bool = False):  # pragma: no cover - fallback
        def _decorator(function):
            return function

        return _decorator

    return _passthrough


_streamlit_singleton = _resolve_streamlit_singleton()

_PRELOAD_WORKER: ModuleType | None
try:
    _PRELOAD_WORKER = importlib.import_module("services.preload_worker")
except Exception as preload_exc:  # pragma: no cover - defensive guard
    logger.warning("Lazy preload skipped on startup: %s", preload_exc)
    _PRELOAD_WORKER = None

_POST_LOGIN_INIT_STARTED = False
_POST_LOGIN_INIT_LOCK = threading.Lock()
_POST_LAZY_LOGGED_KEY = "_startup_logged_post_lazy"
_UI_STARTUP_METRIC_KEY = "ui_startup_load_ms"
_UI_STARTUP_LOGGED_KEY = "_ui_startup_logged"
_UI_STARTUP_REPORTED_KEY = "_ui_startup_metric_reported"
_SCIENTIFIC_PRELOAD_RESUMED_KEY = "_scientific_preload_resumed"
_SINGLETON_METRIC_KEY = "startup.singleton_init_ms"
_LAZY_IMPORTS_METRIC_KEY = "startup.lazy_imports_ms"

_singleton_total_ms = 0.0
_lazy_imports_total_ms = 0.0
_instrumentation_flushed = False


def _update_session_metric(key: str, value: float) -> None:
    try:
        st.session_state[key] = float(value)
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo actualizar la métrica de startup %s", key, exc_info=True)


def _record_startup_metric(field: str, value: float) -> None:
    try:
        from shared import qa_profiler  # type: ignore
    except Exception:  # pragma: no cover - qa profiler puede no estar disponible
        return
    recorder = getattr(qa_profiler, "record_startup_metric", None)
    if callable(recorder):
        try:
            recorder(field, float(value))
        except Exception:  # pragma: no cover - la telemetría no debe romper el flujo
            logger.debug("No se pudo registrar la métrica %s en QA", field, exc_info=True)


def record_singleton_initialization(label: str, duration_ms: float) -> None:
    global _singleton_total_ms
    elapsed = max(float(duration_ms), 0.0)
    if elapsed <= 0.0:
        return
    _singleton_total_ms += elapsed
    log_startup_event(
        f"startup.singleton_init | name={label} | duration_ms={elapsed:.2f} | total_ms={_singleton_total_ms:.2f}"
    )
    _update_session_metric(_SINGLETON_METRIC_KEY, _singleton_total_ms)
    _record_startup_metric(_SINGLETON_METRIC_KEY, _singleton_total_ms)


def _record_lazy_import(label: str, duration_ms: float) -> None:
    global _lazy_imports_total_ms
    elapsed = max(float(duration_ms), 0.0)
    if elapsed <= 0.0:
        return
    _lazy_imports_total_ms += elapsed
    log_startup_event(
        f"startup.lazy_import | module={label} | duration_ms={elapsed:.2f} | total_ms={_lazy_imports_total_ms:.2f}"
    )
    _update_session_metric(_LAZY_IMPORTS_METRIC_KEY, _lazy_imports_total_ms)
    _record_startup_metric(_LAZY_IMPORTS_METRIC_KEY, _lazy_imports_total_ms)


@lru_cache(maxsize=None)
def lazy_module(name: str):
    start = time.perf_counter()
    module = importlib.import_module(name)
    duration_ms = (time.perf_counter() - start) * 1000.0
    _record_lazy_import(name, duration_ms)
    return module


@lru_cache(maxsize=None)
def lazy_attr(module: str, attr: str):
    return getattr(lazy_module(module), attr)


def startup_singleton(name: str, *, show_spinner: bool = False):
    def _decorator(factory):
        @_streamlit_singleton(show_spinner=show_spinner)
        def _wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = factory(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000.0
            record_singleton_initialization(name, duration_ms)
            return result

        return _wrapper

    return _decorator


def start_preload_worker(*, paused: bool = True, libraries: Iterable[str] | None = None) -> bool:
    if _PRELOAD_WORKER is None:
        return False
    try:
        return _PRELOAD_WORKER.start_preload_worker(libraries=libraries, paused=paused)
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("No se pudo iniciar el preload worker", exc_info=True)
        return False


def resume_preload_worker(
    *, delay_seconds: float = 0.0, libraries: Iterable[str] | None = None
) -> bool:
    if _PRELOAD_WORKER is None:
        return False
    try:
        return _PRELOAD_WORKER.resume_preload_worker(
            delay_seconds=delay_seconds, libraries=libraries
        )
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("No se pudo reanudar el preload worker", exc_info=True)
        return False


def is_preload_complete() -> bool:
    if _PRELOAD_WORKER is None:
        return False
    try:
        return bool(_PRELOAD_WORKER.is_preload_complete())
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("No se pudo consultar el estado del preload", exc_info=True)
        return False


def record_startup_checkpoint(label: str) -> float:
    elapsed = max((time.perf_counter() - TOTAL_LOAD_START) * 1000.0, 0.0)
    log_startup_event(f"startup_checkpoint | label={label} | startup_load_ms={elapsed:.2f}")
    return elapsed


def record_stage_lazy(*args, **kwargs) -> None:
    try:
        record_stage = lazy_attr("services.performance_timer", "record_stage")
    except Exception:
        logger.debug("No se pudo importar record_stage", exc_info=True)
        return
    try:
        record_stage(*args, **kwargs)
    except Exception:
        logger.debug("No se pudo registrar ui_total_load", exc_info=True)


def record_post_lazy_checkpoint(total_ms: float) -> None:
    try:
        if st.session_state.get(_POST_LAZY_LOGGED_KEY):
            return
    except Exception:
        pass
    log_startup_event(
        f"startup_checkpoint | label=post_lazy_imports | startup_load_ms={float(total_ms):.2f}"
    )
    try:
        st.session_state[_POST_LAZY_LOGGED_KEY] = True
    except Exception:
        logger.debug("No se pudo marcar %s", _POST_LAZY_LOGGED_KEY, exc_info=True)


def update_ui_total_load_metric_lazy(total_ms: float | int | None) -> None:
    try:
        update_metric = lazy_attr("services.performance_timer", "update_ui_total_load_metric")
    except Exception:
        logger.debug("No se pudo importar update_ui_total_load_metric", exc_info=True)
        return
    try:
        update_metric(total_ms)
    except Exception:
        logger.debug("No se pudo actualizar ui_startup_load_ms", exc_info=True)


def update_ui_startup_metric_lazy(total_ms: float | int | None) -> None:
    try:
        update_metric = lazy_attr("services.performance_timer", "update_ui_startup_load_metric")
    except Exception:
        logger.debug("No se pudo importar update_ui_startup_load_metric", exc_info=True)
        return
    try:
        update_metric(total_ms)
    except Exception:
        logger.debug("No se pudo actualizar ui_startup_load_ms", exc_info=True)


def schedule_scientific_preload_resume(*, delay_seconds: float = 0.0) -> None:
    try:
        if st.session_state.get(_SCIENTIFIC_PRELOAD_RESUMED_KEY):
            return
    except Exception:
        logger.debug("No se pudo verificar _scientific_preload_resumed", exc_info=True)
    resumed = resume_preload_worker(delay_seconds=delay_seconds)
    if not resumed and not is_preload_complete():
        return
    try:
        st.session_state[_SCIENTIFIC_PRELOAD_RESUMED_KEY] = True
    except Exception:
        logger.debug("No se pudo marcar _scientific_preload_resumed", exc_info=True)


def record_ui_startup_metric() -> None:
    try:
        if st.session_state.get(_UI_STARTUP_LOGGED_KEY):
            return
    except Exception:
        pass

    elapsed = max((time.perf_counter() - TOTAL_LOAD_START) * 1000.0, 0.0)

    try:
        st.session_state[_UI_STARTUP_METRIC_KEY] = float(elapsed)
    except Exception:
        logger.debug("No se pudo persistir ui_startup_load_ms", exc_info=True)
    try:
        st.session_state[_UI_STARTUP_REPORTED_KEY] = False
    except Exception:
        logger.debug("No se pudo marcar ui_startup_metric_reported", exc_info=True)
    try:
        st.session_state[_UI_STARTUP_LOGGED_KEY] = True
    except Exception:
        logger.debug("No se pudo marcar _ui_startup_logged", exc_info=True)

    try:
        log_startup_event(f"ui_startup_load | value_ms={elapsed:.2f}")
    except Exception:
        logger.debug(
            "No se pudo registrar ui_startup_load en startup_logger",
            exc_info=True,
        )


def flush_startup_instrumentation_metrics() -> None:
    global _instrumentation_flushed
    if _instrumentation_flushed:
        return
    _instrumentation_flushed = True

    if _singleton_total_ms > 0.0:
        record_stage_lazy(
            "startup.singleton_init",
            total_ms=_singleton_total_ms,
            status="success",
        )
        log_startup_event(f"startup.singleton_init_total | value_ms={_singleton_total_ms:.2f}")

    if _lazy_imports_total_ms > 0.0:
        record_stage_lazy(
            "startup.lazy_imports",
            total_ms=_lazy_imports_total_ms,
            status="success",
        )
        log_startup_event(f"startup.lazy_imports_total | value_ms={_lazy_imports_total_ms:.2f}")


def flush_ui_startup_metric(startup_ms: float | int | None) -> None:
    if startup_ms is None:
        return

    try:
        if st.session_state.get(_UI_STARTUP_REPORTED_KEY):
            return
    except Exception:
        pass

    try:
        value = float(startup_ms)
    except Exception:
        logger.debug("Valor ui_startup_load_ms inválido", exc_info=True)
        return

    record_stage_lazy("ui_startup_load", total_ms=value, status="success")
    update_ui_startup_metric_lazy(value)

    try:
        st.session_state[_UI_STARTUP_REPORTED_KEY] = True
    except Exception:
        logger.debug("No se pudo marcar ui_startup_metric_reported", exc_info=True)


def _run_initialization_stage(label: str, action: Callable[[], None]) -> None:
    start = time.perf_counter()
    try:
        action()
    except Exception as exc:
        log_startup_event(f"post_login_init | stage={label} | status=error | error={exc!r}")
        logger.debug("Post login init stage %s failed", label, exc_info=True)
    else:
        elapsed = (time.perf_counter() - start) * 1000.0
        log_startup_event(
            f"post_login_init | stage={label} | status=success | duration_ms={elapsed:.2f}"
        )


def _post_login_initialization_worker() -> None:
    log_startup_event("post_login_init | status=started")

    def _init_sqlite_maintenance() -> None:
        module = lazy_module("services.maintenance")
        config_cls = getattr(module, "SQLiteMaintenanceConfiguration")
        configure = getattr(module, "configure_sqlite_maintenance")
        ensure = getattr(module, "ensure_sqlite_maintenance_started")
        configure(
            config_cls(
                interval_hours=sqlite_maintenance_interval_hours,
                size_threshold_mb=sqlite_maintenance_size_threshold_mb,
                performance_store_ttl_days=performance_store_ttl_days,
                enable_prometheus=enable_prometheus,
            )
        )
        ensure()

    def _init_system_diagnostics() -> None:
        module = lazy_module("services.system_diagnostics")
        config_cls = getattr(module, "SystemDiagnosticsConfiguration")
        configure = getattr(module, "configure_system_diagnostics")
        ensure = getattr(module, "ensure_system_diagnostics_started")
        configure(config_cls())
        ensure()

    def _init_performance_metrics() -> None:
        module = lazy_module("services.performance_timer")
        init_metrics = getattr(module, "init_metrics", None)
        if callable(init_metrics):
            init_metrics()

    _run_initialization_stage("sqlite_maintenance", _init_sqlite_maintenance)
    _run_initialization_stage("system_diagnostics", _init_system_diagnostics)
    _run_initialization_stage("performance_metrics", _init_performance_metrics)

    log_startup_event("post_login_init | status=completed")


def schedule_post_login_initialization() -> None:
    global _POST_LOGIN_INIT_STARTED
    if _POST_LOGIN_INIT_STARTED:
        return
    with _POST_LOGIN_INIT_LOCK:
        if _POST_LOGIN_INIT_STARTED:
            return
        thread = threading.Thread(
            target=_post_login_initialization_worker,
            name="post-login-init",
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            logger.debug("No se pudo iniciar la inicialización post-login", exc_info=True)
            return
        _POST_LOGIN_INIT_STARTED = True


__all__ = [
    "flush_ui_startup_metric",
    "flush_startup_instrumentation_metrics",
    "is_preload_complete",
    "lazy_attr",
    "lazy_module",
    "record_post_lazy_checkpoint",
    "record_stage_lazy",
    "record_startup_checkpoint",
    "record_singleton_initialization",
    "record_ui_startup_metric",
    "resume_preload_worker",
    "schedule_post_login_initialization",
    "schedule_scientific_preload_resume",
    "startup_singleton",
    "start_preload_worker",
    "update_ui_startup_metric_lazy",
    "update_ui_total_load_metric_lazy",
]
