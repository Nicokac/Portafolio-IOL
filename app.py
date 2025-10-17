# app.py
# Orquestación Streamlit + módulos

from __future__ import annotations

import argparse
import html
import importlib
import importlib.util
import json
import logging
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

import streamlit as st

from shared import skeletons

logger = logging.getLogger(__name__)

try:
    _PRELOAD_WORKER = importlib.import_module("services.preload_worker")
except Exception as preload_exc:  # pragma: no cover - defensive guard
    logger.warning("Lazy preload skipped on startup: %s", preload_exc)
    _PRELOAD_WORKER = None


def start_preload_worker(
    *, paused: bool = False, libraries: Iterable[str] | None = None
) -> bool:
    if _PRELOAD_WORKER is None:
        return False
    try:
        return _PRELOAD_WORKER.start_preload_worker(
            libraries=libraries, paused=paused
        )
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


_TOTAL_LOAD_START = time.perf_counter()
skeletons.initialize(_TOTAL_LOAD_START)
logger.info("🧩 Skeleton system initialized at %.2f", _TOTAL_LOAD_START)

_LOGIN_PHASE_START_KEY = "_login_phase_started_at"
_LOGIN_PRELOAD_RECORDED_KEY = "_login_preload_recorded"

if not hasattr(st, "stop"):
    st.stop = lambda: None  # type: ignore[attr-defined]

if not hasattr(st, "container"):
    st.container = lambda *_, **__: nullcontext()  # type: ignore[attr-defined]

if not hasattr(st, "columns"):
    def _dummy_columns(spec: Sequence[int] | int | None = None, *args, **kwargs):
        if isinstance(spec, int):
            count = max(spec, 1)
        elif isinstance(spec, Sequence):
            count = max(len(spec), 1)
        else:
            count = 2
        return tuple(nullcontext() for _ in range(count))

    st.columns = _dummy_columns  # type: ignore[attr-defined]

from services.startup_logger import (
    log_startup_event,
    log_startup_exception,
    log_ui_total_load_metric,
)
log_startup_event("Streamlit app bootstrap initiated")

from shared.config import configure_logging, ensure_tokens_key
from shared.favorite_symbols import FavoriteSymbols
from shared.security_env_validator import validate_security_environment
from shared.settings import (
    enable_prometheus,
    performance_store_ttl_days,
    sqlite_maintenance_interval_hours,
    sqlite_maintenance_size_threshold_mb,
)
from shared.time_provider import TimeProvider
from shared.telemetry import log_default_telemetry
from ui.ui_settings import init_ui, render_ui_controls
from ui.header import render_header
from ui.actions import render_action_menu
from ui.health_sidebar import render_health_monitor_tab, summarize_health_status
from ui.login import render_login_page
from ui.footer import render_footer
from services.cache import get_fx_rates_cached
from controllers.auth import LOGIN_AUTH_TIMESTAMP_KEY, build_iol_client
from services.health import get_health_metrics, record_dependency_status
from ui.helpers.preload import ensure_scientific_preload_ready


analysis_logger = logging.getLogger("analysis")
ANALYSIS_LOG_PATH = Path(__file__).resolve().parent / "analysis.log"

_POST_LOGIN_INIT_STARTED = False
_POST_LOGIN_INIT_LOCK = threading.Lock()
_PRE_LAZY_LOGGED_KEY = "_startup_logged_pre_lazy"
_POST_LAZY_LOGGED_KEY = "_startup_logged_post_lazy"
_UI_STARTUP_METRIC_KEY = "ui_startup_load_ms"
_UI_STARTUP_LOGGED_KEY = "_ui_startup_logged"
_UI_STARTUP_REPORTED_KEY = "_ui_startup_metric_reported"
_SCIENTIFIC_PRELOAD_READY_KEY = "scientific_preload_ready"
_SCIENTIFIC_PRELOAD_RESUMED_KEY = "_scientific_preload_resumed"


@lru_cache(maxsize=None)
def _lazy_module(name: str):
    return importlib.import_module(name)


@lru_cache(maxsize=None)
def _lazy_attr(module: str, attr: str):
    return getattr(_lazy_module(module), attr)


def _record_startup_checkpoint(label: str) -> float:
    elapsed = max((time.perf_counter() - _TOTAL_LOAD_START) * 1000.0, 0.0)
    log_startup_event(
        f"startup_checkpoint | label={label} | startup_load_ms={elapsed:.2f}"
    )
    return elapsed


def _maybe_log_pre_login_checkpoint() -> None:
    try:
        if st.session_state.get(_PRE_LAZY_LOGGED_KEY):
            return
    except Exception:  # pragma: no cover - defensive guard
        return
    elapsed = _record_startup_checkpoint("before_lazy_imports")
    try:
        st.session_state[_PRE_LAZY_LOGGED_KEY] = True
        st.session_state["startup_load_ms_before_lazy"] = elapsed
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("No se pudo persistir startup_load_ms_before_lazy", exc_info=True)


def _mark_scientific_preload_pending() -> None:
    try:
        st.session_state[_SCIENTIFIC_PRELOAD_READY_KEY] = False
    except Exception:  # pragma: no cover - session state may be read-only in tests
        logger.debug("No se pudo marcar scientific_preload_ready", exc_info=True)


def _render_login_phase() -> None:
    _maybe_log_pre_login_checkpoint()
    _mark_scientific_preload_pending()
    try:
        st.session_state.setdefault(_LOGIN_PHASE_START_KEY, time.perf_counter())
        st.session_state.pop(_LOGIN_PRELOAD_RECORDED_KEY, None)
    except Exception:  # pragma: no cover - session state may be read-only in tests
        logger.debug("No se pudo inicializar el seguimiento de login", exc_info=True)
    start_preload_worker(paused=True)
    try:
        render_login_page()
    finally:
        _record_ui_startup_metric()
    st.stop()


def _record_post_lazy_checkpoint(total_ms: float) -> None:
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
        st.session_state["startup_load_ms_after_lazy"] = float(total_ms)
    except Exception:
        logger.debug("No se pudo persistir startup_load_ms_after_lazy", exc_info=True)


def _record_stage_lazy(*args, **kwargs) -> None:
    try:
        record_stage = _lazy_attr("services.performance_timer", "record_stage")
    except Exception:
        logger.debug("No se pudo importar record_stage", exc_info=True)
        return
    try:
        record_stage(*args, **kwargs)
    except Exception:
        logger.debug("No se pudo registrar ui_total_load", exc_info=True)


def _record_login_preload_timings(preload_ready: bool) -> None:
    """Measure login + scientific preload durations once per authentication."""

    try:
        if st.session_state.get(_LOGIN_PRELOAD_RECORDED_KEY):
            return
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("No se pudo leer el estado de métricas de login", exc_info=True)
        return

    try:
        start_value = st.session_state.get(_LOGIN_PHASE_START_KEY)
    except Exception:
        logger.debug("No se pudo acceder al inicio de sesión", exc_info=True)
        return

    if not isinstance(start_value, (int, float)):
        return

    end = time.perf_counter()
    total_ms = max((end - float(start_value)) * 1000.0, 0.0)
    auth_ts = st.session_state.get(LOGIN_AUTH_TIMESTAMP_KEY)
    login_ms: float | None = None
    if isinstance(auth_ts, (int, float)) and auth_ts >= start_value:
        login_ms = max((float(auth_ts) - float(start_value)) * 1000.0, 0.0)

    extra: dict[str, object] = {"preload_ready": bool(preload_ready)}
    if login_ms is not None:
        extra["login_ms"] = f"{login_ms:.2f}"

    _record_stage_lazy(
        "login_preload_total",
        total_ms=total_ms,
        status="success",
        extra=extra,
    )

    if login_ms is not None:
        _record_stage_lazy("login_phase", total_ms=login_ms, status="success")
        preload_ms = max(total_ms - login_ms, 0.0)
        _record_stage_lazy(
            "scientific_preload",
            total_ms=preload_ms,
            status="success",
        )

    try:
        st.session_state[_LOGIN_PRELOAD_RECORDED_KEY] = True
        st.session_state.pop(_LOGIN_PHASE_START_KEY, None)
        st.session_state.pop(LOGIN_AUTH_TIMESTAMP_KEY, None)
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("No se pudo limpiar el estado de métricas de login", exc_info=True)


def _update_ui_total_load_metric_lazy(total_ms: float | int | None) -> None:
    try:
        update_metric = _lazy_attr(
            "services.performance_timer", "update_ui_total_load_metric"
        )
    except Exception:
        logger.debug("No se pudo importar update_ui_total_load_metric", exc_info=True)
        return
    try:
        update_metric(total_ms)
    except Exception:
        logger.debug("No se pudo actualizar ui_startup_load_ms", exc_info=True)


def _update_ui_startup_metric_lazy(total_ms: float | int | None) -> None:
    try:
        update_metric = _lazy_attr(
            "services.performance_timer", "update_ui_startup_load_metric"
        )
    except Exception:
        logger.debug("No se pudo importar update_ui_startup_load_metric", exc_info=True)
        return
    try:
        update_metric(total_ms)
    except Exception:
        logger.debug("No se pudo actualizar ui_startup_load_ms", exc_info=True)


def _schedule_scientific_preload_resume() -> None:
    try:
        if st.session_state.get(_SCIENTIFIC_PRELOAD_RESUMED_KEY):
            return
    except Exception:
        logger.debug("No se pudo verificar _scientific_preload_resumed", exc_info=True)
    resumed = resume_preload_worker(delay_seconds=0.5)
    if not resumed and not is_preload_complete():
        return
    try:
        st.session_state[_SCIENTIFIC_PRELOAD_RESUMED_KEY] = True
    except Exception:
        logger.debug("No se pudo marcar _scientific_preload_resumed", exc_info=True)


def _record_ui_startup_metric() -> None:
    try:
        if st.session_state.get(_UI_STARTUP_LOGGED_KEY):
            return
    except Exception:
        pass

    elapsed = max((time.perf_counter() - _TOTAL_LOAD_START) * 1000.0, 0.0)

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
        log_startup_event(
            f"ui_startup_load | value_ms={elapsed:.2f}"
        )
    except Exception:
        logger.debug("No se pudo registrar ui_startup_load en startup_logger", exc_info=True)


def _flush_ui_startup_metric(startup_ms: float | int | None) -> None:
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

    _record_stage_lazy("ui_startup_load", total_ms=value, status="success")
    _update_ui_startup_metric_lazy(value)

    try:
        st.session_state[_UI_STARTUP_REPORTED_KEY] = True
    except Exception:
        logger.debug("No se pudo marcar ui_startup_metric_reported", exc_info=True)


def _run_initialization_stage(label: str, action: Callable[[], None]) -> None:
    start = time.perf_counter()
    try:
        action()
    except Exception as exc:
        log_startup_event(
            f"post_login_init | stage={label} | status=error | error={exc!r}"
        )
        logger.debug("Post login init stage %s failed", label, exc_info=True)
    else:
        elapsed = (time.perf_counter() - start) * 1000.0
        log_startup_event(
            f"post_login_init | stage={label} | status=success | duration_ms={elapsed:.2f}"
        )


def _post_login_initialization_worker() -> None:
    log_startup_event("post_login_init | status=started")

    def _init_sqlite_maintenance() -> None:
        module = _lazy_module("services.maintenance")
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
        module = _lazy_module("services.system_diagnostics")
        config_cls = getattr(module, "SystemDiagnosticsConfiguration")
        configure = getattr(module, "configure_system_diagnostics")
        ensure = getattr(module, "ensure_system_diagnostics_started")
        configure(config_cls())
        ensure()

    def _init_performance_metrics() -> None:
        module = _lazy_module("services.performance_timer")
        init_metrics = getattr(module, "init_metrics", None)
        if callable(init_metrics):
            init_metrics()

    _run_initialization_stage("sqlite_maintenance", _init_sqlite_maintenance)
    _run_initialization_stage("system_diagnostics", _init_system_diagnostics)
    _run_initialization_stage("performance_metrics", _init_performance_metrics)

    log_startup_event("post_login_init | status=completed")


def _schedule_post_login_initialization() -> None:
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
        try:
            st.session_state["_post_login_init_started"] = True
        except Exception:
            logger.debug("No se pudo marcar _post_login_init_started", exc_info=True)


def _format_total_load_value(total_ms: int) -> str:
    return f"{int(total_ms):,}".replace(",", " ")


def _render_total_load_indicator(placeholder) -> None:
    try:
        elapsed_ms = max(int((time.perf_counter() - _TOTAL_LOAD_START) * 1000), 0)
    except Exception:
        logger.debug("No se pudo calcular el tiempo total de carga", exc_info=True)
        try:
            log_ui_total_load_metric(None)
        except Exception:
            logger.debug(
                "No se pudo registrar ui_total_load en el startup logger",
                exc_info=True,
            )
        return

    try:
        st.session_state["total_load_ms"] = elapsed_ms
    except Exception:
        logger.debug("No se pudo persistir total_load_ms en session_state", exc_info=True)

    try:
        timings = st.session_state.get("portfolio_stage_timings")
        if isinstance(timings, dict) and "total_ms" not in timings:
            timings["total_ms"] = float(elapsed_ms)
    except Exception:
        logger.debug("No se pudo extender portfolio_stage_timings con total_ms", exc_info=True)

    profile_block_durations: list[float] = []
    try:
        stage_timings = st.session_state.get("portfolio_stage_timings")
    except Exception:
        logger.debug("No se pudo acceder a portfolio_stage_timings", exc_info=True)
        stage_timings = None

    if isinstance(stage_timings, dict):
        for key, value in stage_timings.items():
            if key == "total_ms":
                continue
            if value is None:
                continue
            try:
                duration = float(value)
            except (TypeError, ValueError):
                continue
            if duration < 0:
                continue
            profile_block_durations.append(duration)

    logic_total_ms: float | None = None
    if profile_block_durations:
        logic_total_ms = float(sum(profile_block_durations))

    overhead_ms: float | None = None
    if logic_total_ms is not None:
        overhead_ms = max(float(elapsed_ms) - logic_total_ms, 0.0)
    try:
        st.session_state["streamlit_overhead_ms"] = overhead_ms
    except Exception:
        logger.debug("No se pudo almacenar streamlit_overhead_ms", exc_info=True)

    try:
        startup_ms = st.session_state.get(_UI_STARTUP_METRIC_KEY)
    except Exception:
        logger.debug("No se pudo obtener ui_startup_load_ms", exc_info=True)
        startup_ms = None

    _flush_ui_startup_metric(startup_ms)

    formatted_value = _format_total_load_value(elapsed_ms)
    block = (
        "<div class='load-time-indicator'>"
        f"🕒 Tiempo total de carga: {formatted_value} ms"
        "</div>"
    )
    try:
        if placeholder is not None:
            placeholder.markdown(block, unsafe_allow_html=True)
        else:
            st.markdown(block, unsafe_allow_html=True)
    except Exception:
        logger.debug("No se pudo renderizar el indicador de tiempo total", exc_info=True)

    extra_payload: dict[str, object] = {}
    if logic_total_ms is not None:
        extra_payload["profile_block_total_ms"] = round(logic_total_ms, 2)
        extra_payload["profile_block_count"] = len(profile_block_durations)
    if overhead_ms is not None:
        extra_payload["streamlit_overhead_ms"] = round(overhead_ms, 2)
    skeleton_ms, skeleton_label = skeletons.get_metric()
    if skeleton_ms is not None:
        extra_payload["skeleton_render_ms"] = round(float(skeleton_ms), 2)
        extra_payload["ui_first_paint_ms"] = round(float(skeleton_ms), 2)
        if skeleton_label:
            extra_payload["skeleton_placeholder"] = skeleton_label
    if not extra_payload:
        extra_payload = None

    record_kwargs = {"total_ms": elapsed_ms, "status": "success"}
    if extra_payload:
        record_kwargs["extra"] = extra_payload

    _record_stage_lazy("ui_total_load", **record_kwargs)
    _update_ui_total_load_metric_lazy(elapsed_ms)
    try:
        log_ui_total_load_metric(elapsed_ms)
    except Exception:
        logger.debug(
            "No se pudo registrar ui_total_load en el startup logger",
            exc_info=True,
        )
    _record_post_lazy_checkpoint(elapsed_ms)

# Configuración de UI centralizada (tema y layout)
validate_security_environment()
init_ui()


def _inject_tab_animation_support() -> None:
    """Ensure tab panels receive attributes to drive CSS animations."""

    if st.session_state.get("_tab_animation_hook_injected"):
        return

    st.session_state["_tab_animation_hook_injected"] = True

    from streamlit.components.v1 import html

    html(
        """
        <script>
        (function () {
            const rootDoc = window.parent && window.parent.document ? window.parent.document : null;
            if (!rootDoc) {
                return;
            }

            if (rootDoc.body && rootDoc.body.dataset.tabAnimationHook === "ready") {
                return;
            }

            if (rootDoc.body) {
                rootDoc.body.dataset.tabAnimationHook = "ready";
            }

            const updatePanel = (panel) => {
                if (!panel) {
                    return;
                }

                const isHidden = panel.getAttribute("aria-hidden") === "true";
                panel.setAttribute("data-tab-visible", isHidden ? "false" : "true");

                if (!isHidden) {
                    panel.classList.remove("tab-panel--fade-expand");
                    void panel.offsetWidth;
                    panel.classList.add("tab-panel--fade-expand");
                }
            };

            const panelObserver = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    if (mutation.type === "attributes" && mutation.attributeName === "aria-hidden") {
                        updatePanel(mutation.target);
                    }
                });
            });

            const bindPanel = (panel) => {
                if (!panel || panel.dataset.tabAnimationBound === "true") {
                    return;
                }

                panel.dataset.tabAnimationBound = "true";
                updatePanel(panel);
                panelObserver.observe(panel, { attributes: true, attributeFilter: ["aria-hidden"] });
            };

            const scanPanels = () => {
                const panels = rootDoc.querySelectorAll('div[data-baseweb="tab-panel"]');
                panels.forEach(bindPanel);
            };

            const treeObserver = new MutationObserver(scanPanels);
            treeObserver.observe(rootDoc.body, { childList: true, subtree: true });

            scanPanels();
        })();
        </script>
        """,
        height=0,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments relevant for logging."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--log-level", dest="log_level")
    parser.add_argument("--log-format", dest="log_format", choices=["text", "json"])
    args, _ = parser.parse_known_args(argv)
    return args


def _check_dependency(
    name: str,
    label: str,
    probe: Callable[[], None],
) -> dict[str, str | None]:
    status = "ok"
    detail: str | None = None
    level = logging.INFO
    message = f"✅ Dependencia {label} disponible"

    try:
        probe()
    except ImportError as exc:
        status = "error"
        detail = str(exc)
        level = logging.WARNING
        message = f"⚠️ Falta dependencia crítica: {label}"
    except Exception as exc:  # pragma: no cover - defensive fallback
        status = "warning"
        detail = str(exc)
        level = logging.WARNING
        message = f"⚠️ Dependencia degradada: {label}"

    payload = {
        "event": "dependency.check",
        "dependency": name,
        "status": status,
    }
    if detail:
        payload["detail"] = detail

    analysis_logger.log(level, message, extra={"analysis": payload})
    record_dependency_status(
        name,
        status=status,
        detail=detail,
        label=label,
        source="startup",
    )

    return {"status": status, "detail": detail}


def _format_failure_tooltip_text(last_failure_ts: float | None) -> str | None:
    if last_failure_ts is None:
        return None

    snapshot = TimeProvider.from_timestamp(last_failure_ts)
    if snapshot is None:
        return None

    delta = TimeProvider.now_datetime() - snapshot.moment
    seconds = int(delta.total_seconds())
    if seconds < 0:
        seconds = 0

    if seconds < 60:
        label = "1 segundo" if seconds == 1 else f"{seconds} segundos"
    elif seconds < 3600:
        minutes = seconds // 60
        label = "1 minuto" if minutes == 1 else f"{minutes} minutos"
    elif seconds < 86400:
        hours = seconds / 3600
        if hours.is_integer():
            hours_int = int(hours)
            label = "1 hora" if hours_int == 1 else f"{hours_int} horas"
        else:
            label = f"{hours:.1f} horas"
    else:
        days = seconds / 86400
        if days.is_integer():
            days_int = int(days)
            label = "1 día" if days_int == 1 else f"{days_int} días"
        else:
            label = f"{days:.1f} días"

    return f"Última conexión fallida hace {label} • {snapshot.text}"


def _ensure_kaleido_runtime_safe() -> None:
    try:
        from shared.export import ensure_kaleido_runtime
    except Exception as exc:  # pragma: no cover - defensive, depends on runtime
        raise RuntimeError(f"No se pudo importar utilidades de Kaleido: {exc}") from exc

    try:
        if not ensure_kaleido_runtime():
            raise RuntimeError("Runtime de Kaleido no disponible")
    except Exception as exc:  # pragma: no cover - depends on external runtime
        raise RuntimeError(str(exc)) from exc


def _check_critical_dependencies() -> dict[str, dict[str, str | None]]:
    def _probe_plotly() -> None:
        if importlib.util.find_spec("plotly") is None:
            raise ImportError("plotly no está disponible")

    def _probe_kaleido() -> None:
        if importlib.util.find_spec("kaleido") is None:
            raise ImportError("kaleido no está disponible")

    results: dict[str, dict[str, str | None]] = {}
    results["plotly"] = _check_dependency("plotly", "Plotly", _probe_plotly)
    results["kaleido"] = _check_dependency("kaleido", "Kaleido", _probe_kaleido)
    return results


def main(argv: list[str] | None = None):
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid4().hex
    args = _parse_args(argv or [])
    configure_logging(level=args.log_level, json_format=(args.log_format == "json") if args.log_format else None)
    logger.info("requirements.txt es la fuente autorizada de dependencias.")
    ensure_tokens_key()
    _check_critical_dependencies()

    if st.session_state.get("force_login"):
        _render_login_phase()

    if not st.session_state.get("authenticated"):
        _render_login_phase()

    _schedule_scientific_preload_resume()

    _schedule_post_login_initialization()

    if FavoriteSymbols.STATE_KEY not in st.session_state:
        st.session_state[FavoriteSymbols.STATE_KEY] = set()

    fx_rates, fx_error = get_fx_rates_cached()
    if fx_error:
        st.warning(fx_error)
    render_header(rates=fx_rates)

    health_metrics = get_health_metrics()
    (
        status_icon,
        status_label,
        status_detail,
        status_severity,
        last_failure_ts,
    ) = summarize_health_status(metrics=health_metrics)
    detail_html = (
        f"<span class='health-status-badge__detail'>{status_detail}</span>"
        if status_detail
        else ""
    )
    tooltip_text = _format_failure_tooltip_text(last_failure_ts)
    tooltip_attr = ""
    if tooltip_text:
        safe_tooltip = html.escape(tooltip_text, quote=True)
        tooltip_attr = f" title=\"{safe_tooltip}\" data-tooltip=\"{safe_tooltip}\""
    badge_classes = " ".join(
        [
            "health-status-badge",
            f"health-status-badge--{status_severity or 'unknown'}",
        ]
    )
    status_container = st.container()
    status_container.markdown(
        f"""
        <div class='{badge_classes}'{tooltip_attr}>
            <span class='health-status-badge__pulse' aria-hidden='true'></span>
            <span class='health-status-badge__icon'>{status_icon}</span>
            <span class='health-status-badge__label'>{status_label}</span>
            {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    load_time_placeholder = getattr(st, "empty", lambda: None)()

    main_col = st.container()

    preload_ready = ensure_scientific_preload_ready(main_col)
    _record_login_preload_timings(preload_ready)
    if not preload_ready:
        st.warning(
            "No se pudieron precargar las librerías científicas. "
            "Continuamos con una carga diferida."
        )

    cli = build_iol_client()

    portfolio_module = _lazy_module("controllers.portfolio.portfolio")
    default_view_model_service_factory = getattr(
        portfolio_module, "default_view_model_service_factory"
    )
    default_notifications_service_factory = getattr(
        portfolio_module, "default_notifications_service_factory"
    )
    render_portfolio_ui = _lazy_attr(
        "ui.controllers.portfolio_ui", "render_portfolio_ui"
    )
    render_recommendations_tab = _lazy_attr(
        "ui.tabs.recommendations", "render_recommendations_tab"
    )

    portfolio_section_kwargs = {
        "view_model_service_factory": default_view_model_service_factory,
        "notifications_service_factory": default_notifications_service_factory,
    }

    monitoring_label = "Monitoreo"

    if hasattr(st, "tabs"):
        with main_col:
            tab_labels = ["Portafolio", "Recomendaciones", monitoring_label]
            portfolio_tab, recommendations_tab, monitoring_tab = st.tabs(tab_labels)
            _inject_tab_animation_support()
    else:
        portfolio_tab = main_col
        recommendations_tab = main_col
        monitoring_tab = main_col

    refresh_secs = render_portfolio_ui(
        portfolio_tab,
        cli,
        fx_rates,
        **portfolio_section_kwargs,
    )

    if hasattr(st, "tabs"):
        with recommendations_tab:
            render_recommendations_tab()
        with monitoring_tab:
            render_health_monitor_tab(monitoring_tab, metrics=health_metrics)
    else:
        render_recommendations_tab()
        render_health_monitor_tab(main_col, metrics=health_metrics)

    config_panel = st.sidebar.expander("⚙️ Configuración general", expanded=False)
    with config_panel:
        st.markdown("<div class='control-panel__section'>", unsafe_allow_html=True)
        st.markdown("### 🎛️ Panel de control")
        st.caption("Consulta el estado de la sesión y ejecuta acciones clave.")
        st.markdown(f"**🕒 {TimeProvider.now()}**")
        st.markdown("</div>", unsafe_allow_html=True)

        render_action_menu(container=config_panel)

        st.markdown("<div class='control-panel__section'>", unsafe_allow_html=True)
        render_ui_controls(container=config_panel)
        st.markdown("</div>", unsafe_allow_html=True)
    if not st.session_state.get("iol_startup_metric_logged"):
        login_ts_raw = st.session_state.get("iol_login_ok_ts")
        try:
            login_ts = float(login_ts_raw) if login_ts_raw is not None else None
        except (TypeError, ValueError):
            login_ts = None
        if login_ts is not None:
            render_ts = time.time()
            elapsed_ms = max(int((render_ts - login_ts) * 1000), 0)
            event_name = "startup.render_portfolio_complete"
            payload = {
                "event": event_name,
                "elapsed_ms": elapsed_ms,
                "login_ts": login_ts,
                "render_ts": render_ts,
                "session_id": st.session_state.get("session_id"),
            }
            logger.info(event_name, extra=payload)
            analysis_entry = dict(payload)
            analysis_entry["logged_at"] = TimeProvider.now()
            login_snapshot = TimeProvider.from_timestamp(login_ts)
            render_snapshot = TimeProvider.from_timestamp(render_ts)
            if login_snapshot:
                analysis_entry["login_at"] = login_snapshot.text
            if render_snapshot:
                analysis_entry["render_at"] = render_snapshot.text
            try:
                with ANALYSIS_LOG_PATH.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(analysis_entry, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning("No se pudo escribir analysis.log: %s", exc)
            try:
                dataset_hash = st.session_state.get("portfolio_dataset_hash")
            except Exception:  # pragma: no cover - defensive safeguard
                dataset_hash = None
            try:
                overhead_value = st.session_state.get("streamlit_overhead_ms")
            except Exception:
                overhead_value = None
            skeleton_ms, _skeleton_label = skeletons.get_metric()
            telemetry_extra: dict[str, object] = {}
            if isinstance(overhead_value, (int, float)):
                telemetry_extra["streamlit_overhead_ms"] = round(float(overhead_value), 2)
            if skeleton_ms is not None:
                telemetry_extra["skeleton_render_ms"] = round(float(skeleton_ms), 2)
                telemetry_extra["ui_first_paint_ms"] = round(float(skeleton_ms), 2)
            try:
                log_default_telemetry(
                    phase=event_name,
                    elapsed_s=elapsed_ms / 1000.0,
                    dataset_hash=(str(dataset_hash) if dataset_hash else None),
                    ui_total_load_ms=elapsed_ms,
                    extra=telemetry_extra or None,
                )
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug(
                    "No se pudo registrar telemetría para startup.render_portfolio_complete",
                    exc_info=True,
                )
            st.session_state["iol_startup_metric_logged"] = True

    render_footer()

    _render_total_load_indicator(load_time_placeholder)

    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    try:
        do_refresh = (refresh_secs is not None) and (float(refresh_secs) > 0)
    except (TypeError, ValueError) as e:
        logger.exception("refresh_secs inválido: %s", e)
        do_refresh = True
    if do_refresh and (time.time() - st.session_state["last_refresh"] >= float(refresh_secs)):
        st.session_state["last_refresh"] = time.time()
        st.session_state["show_refresh_toast"] = True
        st.rerun()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])

