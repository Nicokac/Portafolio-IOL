import streamlit as st
from shared.time_provider import TimeProvider
from shared.version import __version__


def get_version() -> str:
    return __version__


def render_footer():
    version = get_version()
    timestamp = TimeProvider.now()
    year = TimeProvider.now_datetime().year
    st.markdown(
        f"""
        <hr>
        <div style='text-align:center; font-size:0.9rem; color:#555;'>
            <div style='font-weight:600; margin-bottom:0.25rem;'>Observabilidad operativa · versión {version}</div>
            <div style='margin-bottom:0.75rem;'>Última sincronización: <strong>{timestamp}</strong></div>
            <div style='font-size:0.8rem; color:#666;'>Desarrollado por Nicolás K. · <a href='https://github.com/caliari' target='_blank'>Portafolio</a></div>
            <div style='font-size:0.75rem; color:#777; margin-top:0.5rem;'>
                &copy; {year}. Los datos se ofrecen sin garantía. Uso bajo su responsabilidad.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
