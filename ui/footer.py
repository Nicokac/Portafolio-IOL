from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from shared.version import __version__


def get_version():
    version = __version__
    now = datetime.now(ZoneInfo("America/Argentina/Buenos_Aires"))

    return version, now


def render_footer():
    version, now = get_version()
    formatted_time = now.strftime("%d/%m/%Y %H:%M:%S")
    year = now.year
    st.markdown(
        f"""
        <hr>
        <div style='text-align:center; font-size:0.9em;'>
            Desarrollado por Nicolás K. ·
            <a href='https://github.com/caliari' target='_blank'>Portafolio</a><br>
            Versión {version} ({formatted_time})<br>
            &copy; {year} - Los datos se ofrecen sin garantía. Uso bajo su responsabilidad.
        </div>
        """,
        unsafe_allow_html=True,
    )

