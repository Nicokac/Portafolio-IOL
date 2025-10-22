import os
import platform
import tempfile

import streamlit as st

from services.update_checker import get_update_history
from shared.version import __version__


def render_about_panel() -> None:
    st.header("ℹ️ Acerca de Portafolio-IOL")
    st.caption(f"Versión: v{__version__}")
    st.caption(f"Sistema operativo: {platform.system()} {platform.release()}")
    st.caption(f"Python: {platform.python_version()}")

    temp_dir = tempfile.gettempdir()
    st.caption(f"Directorio temporal: {temp_dir}")

    with st.expander("📜 Últimos eventos de actualización"):
        history = get_update_history()
        if history:
            for entry in reversed(history[-10:]):
                st.caption(f"🕒 {entry['timestamp']} — {entry['event']} v{entry['version']} ({entry['status']})")
        else:
            st.caption("No se encontraron eventos recientes.")

    with st.expander("🧠 Información del entorno"):
        st.json(
            {
                "cwd": os.getcwd(),
                "environment": dict(os.environ) if os.environ else {},
            }
        )
