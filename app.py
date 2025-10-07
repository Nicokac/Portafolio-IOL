# app.py
# Orquestación Streamlit + módulos

from __future__ import annotations

import argparse
import importlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

import streamlit as st

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

from shared.config import configure_logging, ensure_tokens_key
from shared.settings import FEATURE_OPPORTUNITIES_TAB
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
    render_portfolio_section,
)
from services.cache import get_fx_rates_cached
from controllers.auth import build_iol_client
from services.health import get_health_metrics, record_dependency_status


logger = logging.getLogger(__name__)
analysis_logger = logging.getLogger("analysis")
ANALYSIS_LOG_PATH = Path(__file__).resolve().parent / "analysis.log"

# Configuración de UI centralizada (tema y layout)
init_ui()

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
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.4rem 0.9rem;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.08);
            color: rgb(24, 40, 58);
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 1.75rem;
        }

        .health-status-badge__detail {
            font-weight: 500;
            color: rgba(15, 23, 42, 0.65);
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

        .control-panel__actions .stButton button {
            border-radius: 999px;
            background: rgba(16, 163, 127, 0.12);
            border: 1px solid rgba(16, 163, 127, 0.28);
            color: rgb(11, 83, 69);
        }

        .control-panel__actions .stButton button:hover {
            background: rgba(16, 163, 127, 0.18);
            border-color: rgba(16, 163, 127, 0.35);
            color: rgb(7, 65, 55);
        }

        .control-panel__actions .stButton button:focus {
            box-shadow: 0 0 0 0.2rem rgba(16, 163, 127, 0.25);
        }

        .control-panel__body .stCaption, .control-panel__section .stCaption {
            color: rgba(15, 23, 42, 0.65);
        }

        .control-panel__section + .control-panel__section {
            margin-top: 0.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
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
    status_icon, status_label, status_detail = summarize_health_status(
        metrics=health_metrics
    )
    detail_html = (
        f"<span class='health-status-badge__detail'>{status_detail}</span>"
        if status_detail
        else ""
    )
    status_container = st.container()
    status_container.markdown(
        f"""
        <div class='health-status-badge'>
            <span class='health-status-badge__icon'>{status_icon}</span>
            <span class='health-status-badge__label'>{status_label}</span>
            {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

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
            tab_labels = ["Portafolio", "Empresas con oportunidad", monitoring_label]
            portfolio_tab, opportunities_tab, monitoring_tab = st.tabs(tab_labels)
        refresh_secs = render_portfolio_section(
            portfolio_tab,
            cli,
            fx_rates,
            **portfolio_section_kwargs,
        )
        with opportunities_tab:
            from ui.tabs.opportunities import render_opportunities_tab

            render_opportunities_tab()
        with monitoring_tab:
            render_health_monitor_tab(monitoring_tab, metrics=health_metrics)
    else:
        if hasattr(st, "tabs"):
            with main_col:
                tab_labels = ["Portafolio", monitoring_label]
                portfolio_tab, monitoring_tab = st.tabs(tab_labels)
        else:
            portfolio_tab = main_col
            monitoring_tab = main_col
        refresh_secs = render_portfolio_section(
            portfolio_tab,
            cli,
            fx_rates,
            **portfolio_section_kwargs,
        )
        if FEATURE_OPPORTUNITIES_TAB and not hasattr(st, "tabs"):
            logger.debug("Streamlit stub sin soporte para tabs; se omite pestaña de oportunidades")
        if hasattr(st, "tabs"):
            with monitoring_tab:
                render_health_monitor_tab(monitoring_tab, metrics=health_metrics)
        else:
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

