# app.py
# Orquestación Streamlit + módulos

from __future__ import annotations

import argparse
import html
import importlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

import streamlit as st

_TOTAL_LOAD_START = time.perf_counter()

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
from services.system_diagnostics import (
    SystemDiagnosticsConfiguration,
    configure_system_diagnostics,
    ensure_system_diagnostics_started,
)

from services.performance_timer import record_stage

log_startup_event("Streamlit app bootstrap initiated")

from shared.config import configure_logging, ensure_tokens_key
from shared.security_env_validator import validate_security_environment
from shared.settings import (
    FEATURE_OPPORTUNITIES_TAB,
    enable_prometheus,
    performance_store_ttl_days,
    sqlite_maintenance_interval_hours,
    sqlite_maintenance_size_threshold_mb,
)
try:
    from services.maintenance import (
        SQLiteMaintenanceConfiguration,
        configure_sqlite_maintenance,
        ensure_sqlite_maintenance_started,
    )
except Exception as exc:  # pragma: no cover - exercised via dedicated test
    log_startup_exception(exc)
    print("⚠️ Error capturado, revisar logs/app_startup.log")
    raise
from shared.time_provider import TimeProvider
from ui.ui_settings import init_ui, render_ui_controls
from ui.header import render_header
from ui.actions import render_action_menu
from ui.health_sidebar import render_health_monitor_tab, summarize_health_status
from ui.login import render_login_page
from ui.footer import render_footer
#from controllers.fx import render_fx_section
from controllers.portfolio.portfolio import (
    default_notifications_service_factory,
    default_view_model_service_factory,
)
from services.cache import get_fx_rates_cached
from controllers.auth import build_iol_client
from services.health import get_health_metrics, record_dependency_status
from ui.tabs.recommendations import render_recommendations_tab
from ui.controllers.portfolio_ui import render_portfolio_ui


logger = logging.getLogger(__name__)
analysis_logger = logging.getLogger("analysis")
ANALYSIS_LOG_PATH = Path(__file__).resolve().parent / "analysis.log"


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

    try:
        record_stage("ui_total_load", total_ms=elapsed_ms, status="success")
    except Exception:
        logger.debug("No se pudo registrar ui_total_load en performance_timer", exc_info=True)
    try:
        log_ui_total_load_metric(elapsed_ms)
    except Exception:
        logger.debug(
            "No se pudo registrar ui_total_load en el startup logger",
            exc_info=True,
        )

# Configuración de UI centralizada (tema y layout)
validate_security_environment()
init_ui()
configure_sqlite_maintenance(
    SQLiteMaintenanceConfiguration(
        interval_hours=sqlite_maintenance_interval_hours,
        size_threshold_mb=sqlite_maintenance_size_threshold_mb,
        performance_store_ttl_days=performance_store_ttl_days,
        enable_prometheus=enable_prometheus,
    )
)
ensure_sqlite_maintenance_started()
configure_system_diagnostics(SystemDiagnosticsConfiguration())
ensure_system_diagnostics_started()

st.markdown(
    """
    <style>
        [data-testid="block-container"] {
            max-width: 100%;
            padding: 3.5rem 3.25rem 2.75rem;
            margin: 0 auto;
        }

        @media (max-width: 992px) {
            [data-testid="block-container"] {
                padding: 3rem 2.2rem 2.25rem;
            }
        }

        @media (max-width: 640px) {
            [data-testid="block-container"] {
                padding: 2.4rem 1.3rem 2rem;
            }
        }

        .health-status-badge {
            --badge-bg: rgba(15, 23, 42, 0.08);
            --badge-fg: rgb(24, 40, 58);
            --badge-detail: rgba(15, 23, 42, 0.65);
            --badge-indicator: rgba(16, 163, 127, 0.85);
            --badge-indicator-shadow: rgba(16, 163, 127, 0.35);
            display: inline-flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.45rem 1.1rem 0.45rem 0.95rem;
            border-radius: 999px;
            background: var(--badge-bg);
            color: var(--badge-fg);
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 1.75rem;
            transition: background-color 200ms ease, color 200ms ease;
        }

        .health-status-badge__pulse {
            width: 0.6rem;
            height: 0.6rem;
            border-radius: 50%;
            background: var(--badge-indicator);
            box-shadow: 0 0 0 0 var(--badge-indicator-shadow);
            animation: healthPulse 3.2s ease-in-out infinite;
            flex-shrink: 0;
        }

        .health-status-badge__icon {
            line-height: 1;
        }

        .health-status-badge__detail {
            font-weight: 500;
            color: var(--badge-detail);
        }

        .health-status-badge--success {
            --badge-bg: rgba(16, 163, 127, 0.12);
            --badge-fg: rgb(7, 65, 55);
            --badge-detail: rgba(7, 65, 55, 0.7);
            --badge-indicator: rgba(16, 163, 127, 0.95);
            --badge-indicator-shadow: rgba(16, 163, 127, 0.35);
        }

        .load-time-indicator {
            color: rgba(15, 23, 42, 0.65);
            font-size: 0.9rem;
            margin: -0.75rem 0 1.5rem;
        }

        [data-theme="dark"] .load-time-indicator {
            color: rgba(226, 232, 240, 0.75);
        }

        .health-status-badge--warning {
            --badge-bg: rgba(202, 138, 4, 0.14);
            --badge-fg: rgb(133, 77, 14);
            --badge-detail: rgba(133, 77, 14, 0.75);
            --badge-indicator: rgba(217, 119, 6, 0.92);
            --badge-indicator-shadow: rgba(217, 119, 6, 0.36);
        }

        .health-status-badge--danger {
            --badge-bg: rgba(220, 38, 38, 0.14);
            --badge-fg: rgb(153, 27, 27);
            --badge-detail: rgba(153, 27, 27, 0.75);
            --badge-indicator: rgba(239, 68, 68, 0.92);
            --badge-indicator-shadow: rgba(239, 68, 68, 0.4);
        }

        .health-status-badge--unknown {
            --badge-bg: rgba(100, 116, 139, 0.14);
            --badge-fg: rgb(71, 85, 105);
            --badge-detail: rgba(71, 85, 105, 0.7);
            --badge-indicator: rgba(148, 163, 184, 0.9);
            --badge-indicator-shadow: rgba(148, 163, 184, 0.36);
        }

        @keyframes healthPulse {
            0% {
                box-shadow: 0 0 0 0 var(--badge-indicator-shadow);
                transform: scale(0.98);
            }
            60% {
                box-shadow: 0 0 0 0.9rem rgba(0, 0, 0, 0);
                transform: scale(1);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(0, 0, 0, 0);
                transform: scale(0.98);
            }
        }

        @media (prefers-reduced-motion: reduce) {
            .health-status-badge__pulse {
                animation: none;
            }
        }

        [data-theme="dark"] .health-status-badge {
            --badge-bg: rgba(148, 163, 184, 0.16);
            --badge-fg: rgba(226, 232, 240, 0.92);
            --badge-detail: rgba(226, 232, 240, 0.65);
            --badge-indicator-shadow: rgba(94, 234, 212, 0.28);
        }

        [data-theme="dark"] .health-status-badge--success {
            --badge-bg: rgba(16, 185, 129, 0.22);
            --badge-fg: rgba(190, 242, 100, 0.92);
            --badge-detail: rgba(190, 242, 100, 0.72);
            --badge-indicator: rgba(16, 185, 129, 0.95);
            --badge-indicator-shadow: rgba(16, 185, 129, 0.4);
        }

        [data-theme="dark"] .health-status-badge--warning {
            --badge-bg: rgba(202, 138, 4, 0.28);
            --badge-fg: rgba(253, 224, 71, 0.92);
            --badge-detail: rgba(253, 224, 71, 0.72);
            --badge-indicator: rgba(234, 179, 8, 0.95);
            --badge-indicator-shadow: rgba(234, 179, 8, 0.42);
        }

        [data-theme="dark"] .health-status-badge--danger {
            --badge-bg: rgba(239, 68, 68, 0.28);
            --badge-fg: rgba(252, 165, 165, 0.94);
            --badge-detail: rgba(252, 165, 165, 0.72);
            --badge-indicator: rgba(248, 113, 113, 0.95);
            --badge-indicator-shadow: rgba(248, 113, 113, 0.45);
        }

        [data-theme="dark"] .health-status-badge--unknown {
            --badge-bg: rgba(100, 116, 139, 0.32);
            --badge-fg: rgba(226, 232, 240, 0.85);
            --badge-detail: rgba(226, 232, 240, 0.62);
            --badge-indicator: rgba(148, 163, 184, 0.9);
            --badge-indicator-shadow: rgba(148, 163, 184, 0.36);
        }

        .control-panel__body {
            background: rgba(15, 23, 42, 0.06);
            border-radius: 1.25rem;
            padding: 1.75rem 2rem;
        }

        .control-panel__body--sidebar {
            padding: 1.6rem 1.75rem;
            margin-bottom: 1.6rem;
        }

        .control-panel__section {
            background: rgba(15, 23, 42, 0.04);
            border-radius: 1rem;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.4rem;
        }

        .control-panel__section:last-child {
            margin-bottom: 0;
        }

        .control-panel__banner {
            padding: 0.85rem 1.2rem;
            border-radius: 0.95rem;
            background: rgba(16, 163, 127, 0.12);
            border: 1px solid rgba(16, 163, 127, 0.28);
            color: rgb(7, 65, 55);
            font-weight: 600;
            margin-bottom: 1rem;
        }

        .control-panel__banner--logout {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .control-panel__banner--logout::before {
            content: "\1F512";
        }

        .fade-out {
            animation: fade-out 0.9s ease-in forwards;
            animation-delay: 0.9s;
        }

        @keyframes fade-out {
            0% {
                opacity: 1;
            }

            100% {
                opacity: 0;
            }
        }

        .control-panel__actions .stButton button {
            border-radius: 999px;
            background: rgba(16, 163, 127, 0.12);
            background: color-mix(in srgb, var(--color-accent) 16%, transparent);
            border: 1px solid color-mix(in srgb, var(--color-accent) 36%, transparent);
            color: color-mix(in srgb, var(--color-accent) 78%, var(--color-text) 22%);
            transition: background-color 150ms ease, border-color 150ms ease,
                color 150ms ease, box-shadow 150ms ease, transform 120ms ease;
        }

        .control-panel__actions .stButton button:hover,
        .control-panel__actions .stButton button:focus-visible {
            background: color-mix(in srgb, var(--color-accent) 26%, var(--color-bg) 74%);
            border-color: color-mix(in srgb, var(--color-accent) 48%, transparent);
            color: color-mix(in srgb, var(--color-accent) 86%, var(--color-text) 14%);
        }

        .control-panel__actions .stButton button:active {
            background: color-mix(in srgb, var(--color-accent) 34%, var(--color-bg) 66%);
            border-color: color-mix(in srgb, var(--color-accent) 60%, transparent);
            transform: translateY(1px);
        }

        .control-panel__actions .stButton button:focus-visible {
            outline: none;
            box-shadow: 0 0 0 0.18rem color-mix(in srgb, var(--color-accent) 32%, transparent);
        }

        [data-testid="stSidebar"] [role="switch"] {
            border-radius: 999px;
            background: color-mix(in srgb, var(--color-text) 14%, transparent);
            border: 1px solid color-mix(in srgb, var(--color-text) 22%, transparent);
            transition: background-color 150ms ease, border-color 150ms ease,
                box-shadow 150ms ease;
            outline: none;
        }

        [data-testid="stSidebar"] [role="switch"][aria-checked="true"] {
            background: color-mix(in srgb, var(--color-accent) 40%, var(--color-bg) 60%);
            border-color: color-mix(in srgb, var(--color-accent) 58%, transparent);
        }

        [data-testid="stSidebar"] [role="switch"]:hover {
            border-color: color-mix(in srgb, var(--color-accent) 46%, transparent);
        }

        [data-testid="stSidebar"] [role="switch"]:active {
            background: color-mix(in srgb, var(--color-accent) 48%, var(--color-bg) 52%);
        }

        [data-testid="stSidebar"] [role="switch"]:focus-visible {
            box-shadow: 0 0 0 0.18rem color-mix(in srgb, var(--color-accent) 34%, transparent);
        }

        [data-testid="stSidebar"] div[data-baseweb="select"] {
            border-radius: 0.75rem;
            border: 1px solid color-mix(in srgb, var(--color-text) 22%, transparent);
            transition: border-color 150ms ease, box-shadow 150ms ease,
                background-color 150ms ease;
        }

        [data-testid="stSidebar"] div[data-baseweb="select"]:hover {
            border-color: color-mix(in srgb, var(--color-accent) 48%, transparent);
        }

        [data-testid="stSidebar"] div[data-baseweb="select"]:focus-within {
            border-color: color-mix(in srgb, var(--color-accent) 66%, transparent);
            box-shadow: 0 0 0 0.18rem color-mix(in srgb, var(--color-accent) 30%, transparent);
            background: color-mix(in srgb, var(--color-accent) 12%, var(--color-bg) 88%);
        }

        [data-testid="stSidebar"] div[data-baseweb="select"]:active {
            border-color: color-mix(in srgb, var(--color-accent) 70%, transparent);
        }

        .control-panel__body .stCaption, .control-panel__section .stCaption {
            color: rgba(15, 23, 42, 0.65);
        }

        .control-panel__section + .control-panel__section {
            margin-top: 0.2rem;
        }

        div[data-baseweb="tab-panel"] {
            position: relative;
            transition: opacity 0.18s ease-out, transform 0.22s ease-out;
            will-change: opacity, transform;
        }

        div[data-baseweb="tab-panel"][data-tab-visible="false"] {
            opacity: 0;
            transform: translateY(0.6rem) scale(0.985);
            pointer-events: none;
        }

        div[data-baseweb="tab-panel"][data-tab-visible="true"] {
            opacity: 1;
            transform: translateY(0) scale(1);
            pointer-events: auto;
        }

        div[data-baseweb="tab-panel"].tab-panel--fade-expand {
            animation: tab-fade-expand 0.24s ease-out both;
        }

        @keyframes tab-fade-expand {
            from {
                opacity: 0;
                transform: translateY(0.6rem) scale(0.985);
            }

            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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
        importlib.import_module("plotly")

    def _probe_kaleido() -> None:
        importlib.import_module("kaleido")
        _ensure_kaleido_runtime_safe()

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
        render_login_page()
        st.stop()

    if not st.session_state.get("authenticated"):
        render_login_page()
        st.stop()

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

    cli = build_iol_client()

    can_render_opportunities = FEATURE_OPPORTUNITIES_TAB and hasattr(st, "tabs")

    portfolio_section_kwargs = {
        "view_model_service_factory": default_view_model_service_factory,
        "notifications_service_factory": default_notifications_service_factory,
    }

    monitoring_label = "Monitoreo"

    if can_render_opportunities and hasattr(st, "tabs"):
        with main_col:
            tab_labels = [
                "Portafolio",
                "Recomendaciones",
                "Empresas con oportunidad",
                monitoring_label,
            ]
            (
                portfolio_tab,
                recommendations_tab,
                opportunities_tab,
                monitoring_tab,
            ) = st.tabs(tab_labels)
            _inject_tab_animation_support()
        refresh_secs = render_portfolio_ui(
            portfolio_tab,
            cli,
            fx_rates,
            **portfolio_section_kwargs,
        )
        with recommendations_tab:
            render_recommendations_tab()
        with opportunities_tab:
            from ui.tabs.opportunities import render_opportunities_tab

            render_opportunities_tab()
        with monitoring_tab:
            render_health_monitor_tab(monitoring_tab, metrics=health_metrics)
    else:
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
        if FEATURE_OPPORTUNITIES_TAB and not hasattr(st, "tabs"):
            logger.debug("Streamlit stub sin soporte para tabs; se omite pestaña de oportunidades")
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

