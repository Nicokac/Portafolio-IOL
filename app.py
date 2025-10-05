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
    st.container = lambda: nullcontext()  # type: ignore[attr-defined]

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
from ui.ui_settings import init_ui
from ui.header import render_header
from ui.actions import render_action_menu
from ui.health_sidebar import render_health_sidebar
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
from services.health import record_dependency_status


logger = logging.getLogger(__name__)
analysis_logger = logging.getLogger("analysis")
ANALYSIS_LOG_PATH = Path(__file__).resolve().parent / "analysis.log"

# Configuración de UI centralizada (tema y layout)
init_ui()


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
    content_col, controls_col = st.columns([5, 2], gap="large")
    with controls_col:
        controls_area = st.container()
        with controls_area:
            st.markdown("#### 🎛️ Panel de control")
            st.caption("Consulta el estado de la sesión y ejecuta acciones clave.")
            timestamp = TimeProvider.now()
            st.markdown(f"**🕒 {timestamp}**")
            render_action_menu()
    main_col = content_col.container()

    cli = build_iol_client()

    can_render_opportunities = FEATURE_OPPORTUNITIES_TAB and hasattr(st, "tabs")

    portfolio_section_kwargs = {
        "view_model_service_factory": default_view_model_service_factory,
        "notifications_service_factory": default_notifications_service_factory,
    }

    if can_render_opportunities:
        with main_col:
            tab_labels = ["Portafolio", "Empresas con oportunidad"]
            portfolio_tab, opportunities_tab = st.tabs(tab_labels)
        refresh_secs = render_portfolio_section(
            portfolio_tab,
            cli,
            fx_rates,
            **portfolio_section_kwargs,
        )
        with opportunities_tab:
            from ui.tabs.opportunities import render_opportunities_tab

            render_opportunities_tab()
    else:
        refresh_secs = render_portfolio_section(
            main_col,
            cli,
            fx_rates,
            **portfolio_section_kwargs,
        )
        if FEATURE_OPPORTUNITIES_TAB and not hasattr(st, "tabs"):
            logger.debug("Streamlit stub sin soporte para tabs; se omite pestaña de oportunidades")
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
    render_health_sidebar()

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

