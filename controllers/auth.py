import streamlit as st

from infrastructure.iol.client import IIOLProvider
from services.cache import build_iol_client as _build_iol_client
from infrastructure.cache import cache


def build_iol_client() -> IIOLProvider | None:
    cli, error = _build_iol_client()
    if error:
        cache.session_state["login_error"] = error
        cache.session_state["force_login"] = True
        cache.session_state["IOL_PASSWORD"] = ""
        st.rerun()
        return None
    cache.session_state["authenticated"] = True
    cache.session_state.pop("IOL_PASSWORD", None)
    return cli

