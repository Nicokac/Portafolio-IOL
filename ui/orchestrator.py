"""Main UI orchestration helpers for the Streamlit application."""

from __future__ import annotations

import html
import importlib.util
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import streamlit as st

from bootstrap.config import TOTAL_LOAD_START
from bootstrap.startup import (
    flush_ui_startup_metric,
    lazy_attr,
    lazy_module,
    record_post_lazy_checkpoint,
    record_stage_lazy,
    record_startup_checkpoint,
    record_ui_startup_metric,
    schedule_post_login_initialization,
    schedule_scientific_preload_resume,
    start_preload_worker,
    update_ui_total_load_metric_lazy,
)
from services.cache import get_fx_rates_cached
from services.health import get_health_metrics, record_dependency_status
from services.startup_logger import log_ui_total_load_metric
from shared import skeletons
from shared.favorite_symbols import FavoriteSymbols
from shared.telemetry import log_default_telemetry
from shared.time_provider import TimeProvider
from ui.actions import render_action_menu
from ui.footer import render_footer
from ui.header import render_header
from ui.health_sidebar import render_health_monitor_tab, summarize_health_status
from ui.helpers.preload import ensure_scientific_preload_ready
from ui.login import render_login_page
from ui.ui_settings import render_ui_controls

logger = logging.getLogger("ui.orchestrator")


try:
    from controllers.auth import LOGIN_AUTH_TIMESTAMP_KEY, build_iol_client
except ImportError:  # pragma: no cover - tests may stub controllers.auth
    LOGIN_AUTH_TIMESTAMP_KEY = "login_auth_timestamp"

    def build_iol_client():  # type: ignore[override]
        return None


analysis_logger = logging.getLogger("analysis")
ANALYSIS_LOG_PATH = Path(__file__).resolve().parent.parent / "analysis.log"

_LOGIN_PHASE_START_KEY = "_login_phase_started_at"
_LOGIN_PRELOAD_RECORDED_KEY = "_login_preload_recorded"
_PRE_LAZY_LOGGED_KEY = "_startup_logged_pre_lazy"
_UI_STARTUP_METRIC_KEY = "ui_startup_load_ms"
_UI_STARTUP_REPORTED_KEY = "_ui_startup_metric_reported"
_SCIENTIFIC_PRELOAD_READY_KEY = "scientific_preload_ready"


def _maybe_log_pre_login_checkpoint() -> None:
    try:
        if st.session_state.get(_PRE_LAZY_LOGGED_KEY):
            return
    except Exception:  # pragma: no cover - defensive guard
        return
    elapsed = record_startup_checkpoint("before_lazy_imports")
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
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("No se pudo inicializar el seguimiento de login", exc_info=True)
    start_preload_worker(paused=True)
    try:
        render_login_page()
    finally:
        record_ui_startup_metric()
    st.stop()


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

    record_stage_lazy(
        "login_preload_total",
        total_ms=total_ms,
        status="success",
        extra=extra,
    )

    if login_ms is not None:
        record_stage_lazy("login_phase", total_ms=login_ms, status="success")
        preload_ms = max(total_ms - login_ms, 0.0)
        record_stage_lazy(
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


def _format_total_load_value(total_ms: int) -> str:
    return f"{int(total_ms):,}".replace(",", " ")


def _render_total_load_indicator(placeholder) -> None:
    try:
        elapsed_ms = max(int((time.perf_counter() - TOTAL_LOAD_START) * 1000), 0)
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

    flush_ui_startup_metric(startup_ms)

    formatted_value = _format_total_load_value(elapsed_ms)
    block = f"<div class='load-time-indicator'>🕒 Tiempo total de carga: {formatted_value} ms</div>"
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

    record_stage_lazy("ui_total_load", **record_kwargs)
    update_ui_total_load_metric_lazy(elapsed_ms)
    try:
        log_ui_total_load_metric(elapsed_ms)
    except Exception:
        logger.debug(
            "No se pudo registrar ui_total_load en el startup logger",
            exc_info=True,
        )
    record_post_lazy_checkpoint(elapsed_ms)


def _inject_tab_animation_support() -> None:
    """Ensure tab panels receive attributes to drive CSS animations."""

    if st.session_state.get("_tab_animation_hook_injected"):
        return

    st.session_state["_tab_animation_hook_injected"] = True

    from streamlit.components.v1 import html as component_html

    component_html(
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


def render_main_ui() -> None:
    first_frame_placeholder = getattr(st, "empty", lambda: None)()
    already_rendered = False
    try:
        already_rendered = bool(st.session_state.get("_app_shell_placeholder_rendered"))
    except Exception:
        logger.debug("No se pudo comprobar el estado del skeleton inicial", exc_info=True)
    if not already_rendered:
        shell_container = skeletons.mark_placeholder("app_shell", placeholder=first_frame_placeholder)
        if shell_container is not None:
            try:
                shell_container.markdown("⌛ Preparando tu portafolio…")
            except Exception:
                logger.debug("No se pudo renderizar el skeleton inicial", exc_info=True)
            else:
                try:
                    st.session_state["_app_shell_placeholder_rendered"] = True
                except Exception:
                    logger.debug(
                        "No se pudo marcar el skeleton inicial como renderizado",
                        exc_info=True,
                    )
    try:
        first_paint_metric, _ = skeletons.get_metric()
    except Exception:
        first_paint_metric = None
    if first_paint_metric is not None:
        try:
            st.session_state.setdefault("ui_first_paint_ms", float(first_paint_metric))
        except Exception:
            logger.debug("No se pudo persistir ui_first_paint_ms", exc_info=True)
    _check_critical_dependencies()

    if st.session_state.get("force_login"):
        _render_login_phase()

    if not st.session_state.get("authenticated"):
        _render_login_phase()

    schedule_scientific_preload_resume()

    schedule_post_login_initialization()

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
    detail_html = f"<span class='health-status-badge__detail'>{status_detail}</span>" if status_detail else ""
    tooltip_text = _format_failure_tooltip_text(last_failure_ts)
    tooltip_attr = ""
    if tooltip_text:
        safe_tooltip = html.escape(tooltip_text, quote=True)
        tooltip_attr = f' title="{safe_tooltip}" data-tooltip="{safe_tooltip}"'
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
        st.warning("No se pudieron precargar las librerías científicas. Continuamos con una carga diferida.")

    cli = build_iol_client()

    portfolio_module = lazy_module("controllers.portfolio.portfolio")
    default_view_model_service_factory = getattr(portfolio_module, "default_view_model_service_factory")
    default_notifications_service_factory = getattr(portfolio_module, "default_notifications_service_factory")
    render_portfolio_ui = lazy_attr("ui.controllers.portfolio_ui", "render_portfolio_ui")
    render_recommendations_tab = lazy_attr("ui.tabs.recommendations", "render_recommendations_tab")

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

    if st.session_state.pop("show_refresh_toast", False):
        st.toast("Datos actualizados", icon="✅")
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
            hydration_was_locked = bool(st.session_state.get("_hydration_lock"))
            should_request_rerun = False
            if hydration_was_locked:
                try:
                    st.session_state["_hydration_lock"] = False
                except Exception:  # pragma: no cover - defensive safeguard
                    logger.debug("No se pudo actualizar _hydration_lock", exc_info=True)
                if not st.session_state.get("_hydration_unlock_rerun_triggered"):
                    should_request_rerun = True
                    st.session_state["_hydration_unlock_rerun_triggered"] = True
            try:
                soft_refresh = bool(st.session_state.pop("_soft_refresh_applied", False))
            except Exception:  # pragma: no cover - defensive safeguard
                soft_refresh = False

            payload = {
                "event": event_name,
                "elapsed_ms": elapsed_ms,
                "login_ts": login_ts,
                "render_ts": render_ts,
                "session_id": st.session_state.get("session_id"),
                "soft_refresh": soft_refresh,
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
                    "No se pudo registrar telemetría para startup.render_portfolio_complete",  # noqa: E501
                    exc_info=True,
                )
            st.session_state["iol_startup_metric_logged"] = True
            if should_request_rerun and hasattr(st, "experimental_rerun"):
                st.experimental_rerun()

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


__all__ = ["render_main_ui"]
