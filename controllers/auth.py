import logging
import streamlit as st

from infrastructure.iol.client import IIOLProvider
from services.cache import build_iol_client as _build_iol_client


logger = logging.getLogger(__name__)


def build_iol_client() -> IIOLProvider | None:
    cli, error = _build_iol_client()
    if error:
        logger.exception("build_iol_client failed", exc_info=error)
        st.session_state["login_error"] = "Error de conexión"
        st.session_state["force_login"] = True
        st.session_state["IOL_PASSWORD"] = ""
        st.rerun()
        return None
    st.session_state["authenticated"] = True
    st.session_state.pop("IOL_PASSWORD", None)
    return cli

