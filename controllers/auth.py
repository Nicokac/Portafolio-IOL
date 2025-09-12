import logging
import streamlit as st

from infrastructure.iol.client import IIOLProvider
from application.auth_service import get_auth_provider
from infrastructure.iol.auth import InvalidCredentialsError

logger = logging.getLogger(__name__)


def build_iol_client() -> IIOLProvider | None:
    provider = get_auth_provider()
    cli, error = provider.build_client()
    if error:
        logger.exception("build_iol_client failed", exc_info=error)
        if isinstance(error, InvalidCredentialsError):
            st.session_state["login_error"] = "Credenciales inválidas"
        else:
            st.session_state["login_error"] = "Error de conexión"
        st.session_state["force_login"] = True
        st.session_state["IOL_PASSWORD"] = ""
        st.rerun()
        return None
    st.session_state["authenticated"] = True
    st.session_state.pop("IOL_PASSWORD", None)
    return cli

