# app.py
# Orquestación Streamlit + módulos

from __future__ import annotations

import argparse
import time
import logging
from uuid import uuid4

import streamlit as st

from shared.config import configure_logging, ensure_tokens_key
from shared.time_provider import TimeProvider
from ui.ui_settings import init_ui
from ui.header import render_header
from ui.actions import render_action_menu
from ui.health_sidebar import render_health_sidebar
from ui.login import render_login_page
from ui.footer import render_footer
#from controllers.fx import render_fx_section
from controllers.portfolio import render_portfolio_section
from services.cache import get_fx_rates_cached
from controllers.auth import build_iol_client


logger = logging.getLogger(__name__)

# Configuración de UI centralizada (tema y layout)
init_ui()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments relevant for logging."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--log-level", dest="log_level")
    parser.add_argument("--log-format", dest="log_format", choices=["text", "json"])
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None):
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid4().hex
    args = _parse_args(argv or [])
    configure_logging(level=args.log_level, json_format=(args.log_format == "json") if args.log_format else None)
    ensure_tokens_key()

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
    _, hcol2 = st.columns([4, 1])
    with hcol2:
        st.caption(f"🕒 {TimeProvider.now()}")
        render_action_menu()

    # main_col, side_col = st.columns([4, 1])

    # render_fx_section(side_col, fx_rates)
    main_col = st.container()

    cli = build_iol_client()
    refresh_secs = render_portfolio_section(main_col, cli, fx_rates)
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

