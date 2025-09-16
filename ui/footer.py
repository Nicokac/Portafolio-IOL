import streamlit as st
from shared.version import __version__
from shared.time_provider import TimeProvider


def get_version() -> str:
    return __version__


def render_footer():
    version = get_version()
    now = TimeProvider.now_datetime()
    timestamp = TimeProvider.now()
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

