import subprocess
from datetime import datetime

import streamlit as st
from shared.version import __version__


def get_version():
    version = __version__
    try:
        date = (
            subprocess.check_output(
                ["git", "log", "-1", "--format=%cd", "--date=short"], stderr=subprocess.STDOUT
            )
            .decode()
            .strip()
        )
    except Exception:
        date = "desconocida"

    return version, date


def render_footer():
    version, commit_date = get_version()
    year = datetime.now().year
    st.markdown(
        f"""
        <hr>
        <div style='text-align:center; font-size:0.9em;'>
            Desarrollado por Nicolás K. ·
            <a href='https://github.com/caliari' target='_blank'>Portafolio</a><br>
            Versión {version} ({commit_date})<br>
            &copy; {year} - Los datos se ofrecen sin garantía. Uso bajo su responsabilidad.
        </div>
        """,
        unsafe_allow_html=True,
    )

