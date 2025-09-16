from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from shared.version import __version__

TIMEZONE = "America/Argentina/Buenos_Aires"


def get_version() -> str:
    return __version__


def render_footer():
    version = get_version()
    timezone = ZoneInfo(TIMEZONE)
    now = datetime.now(timezone)
    timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
    year = now.year
    st.markdown(
        f"""
        <hr>
        <div style='text-align:center; font-size:0.9em;'>
            Desarrollado por Nicolás K. ·
            <a href='https://github.com/caliari' target='_blank'>Portafolio</a><br>
            Versión {version}<br>
            Actualizado {timestamp}<br>
            &copy; {year} - Los datos se ofrecen sin garantía. Uso bajo su responsabilidad.
        </div>
        """,
        unsafe_allow_html=True,
    )

