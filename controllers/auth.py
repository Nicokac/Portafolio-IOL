import streamlit as st

from infrastructure.iol.client import IIOLProvider
from services.cache import build_iol_client as _build_iol_client


def build_iol_client() -> IIOLProvider | None:
    cli, error = _build_iol_client()
    if error:
        st.session_state["login_error"] = error
        st.session_state["force_login"] = True
        st.session_state["IOL_PASSWORD"] = ""
        st.rerun()
        return None
    st.session_state["authenticated"] = True
    st.session_state.pop("IOL_PASSWORD", None)
    return cli

