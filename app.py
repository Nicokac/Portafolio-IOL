# app.py
# Orquestación Streamlit + módulos

from __future__ import annotations

import time
from datetime import datetime
import logging

import streamlit as st

from shared.config import settings
from ui.ui_settings import init_ui
from ui.header import render_header
from ui.actions import render_action_menu
from ui.login import render_login_page
from ui.footer import render_footer
#from controllers.fx import render_fx_section
from controllers.portfolio import render_portfolio_section
from services.cache import get_fx_rates_cached, build_iol_client


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuración de UI centralizada (tema y layout)
init_ui()

# Precarga credenciales desde settings en session_state, salvo que se fuerce el login
# o ya esté autenticado. Evitamos persistir la contraseña innecesariamente.
if not st.session_state.get("force_login") and not st.session_state.get("authenticated"):
    if settings.IOL_USERNAME:
        st.session_state.setdefault("IOL_USERNAME", settings.IOL_USERNAME)
    if settings.IOL_PASSWORD:
        st.session_state.setdefault("IOL_PASSWORD", settings.IOL_PASSWORD)


def main():
    if st.session_state.get("force_login"):
        render_login_page()
        st.stop()

    user = st.session_state.get("IOL_USERNAME") or settings.IOL_USERNAME
    password = st.session_state.get("IOL_PASSWORD") or settings.IOL_PASSWORD
    if not st.session_state.get("authenticated"):
        if not user or not password:
            render_login_page()
            st.stop()

    fx_rates, fx_error = get_fx_rates_cached()
    if fx_error:
        st.warning(fx_error)
    render_header(rates=fx_rates)
    _, hcol2 = st.columns([4, 1])
    with hcol2:
        now = datetime.now()
        st.caption(f"🕒 {now.strftime('%d/%m/%Y %H:%M:%S')}")
        render_action_menu()

    # main_col, side_col = st.columns([4, 1])

    # render_fx_section(side_col, fx_rates)
    main_col = st.container()

    cli = build_iol_client()
    # Una vez autenticado, eliminamos la contraseña de la sesión
    st.session_state.pop("IOL_PASSWORD", None)
    refresh_secs = render_portfolio_section(main_col, cli, fx_rates)
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
    main()

