from __future__ import annotations

import streamlit as st

from shared.time_provider import TimeProvider
from shared.version import __version__


def get_version() -> str:
    return __version__


def render_footer() -> None:
    version = get_version()
    timestamp = TimeProvider.now()
    year = TimeProvider.now_datetime().year
    st.markdown(
        f"""
        <hr>
        <div class='footer-container'>
            <div class='footer-column'>
                <div class='footer-title'>Información operativa</div>
                <div class='footer-meta'>
                    <p><strong>Versión:</strong> {version}</p>
                    <p><strong>Última sincronización:</strong> {timestamp}</p>
                    <p>&copy; {year} Portafolio IOL · Datos provistos en modo lectura.</p>
                </div>
            </div>
            <div class='footer-column'>
                <div class='footer-title'>Resumen de release</div>
                <div class='footer-meta'>
                    <p><strong>v0.3.4.3</strong> Layout consolidado y controles unificados.</p>
                    <p>Explorá el tab Monitoreo para revisar el healthcheck completo.</p>
                </div>
            </div>
            <div class='footer-links-card-wrapper'>
                <div class='footer-links-card'>
                    <div class='footer-links-card__title'>Enlaces útiles</div>
                    <div class='footer-links-card__list'>
                        <div>📘 <a href="https://github.com/caliari/Portafolio-IOL#readme" target="_blank" rel="noopener noreferrer">Documentación</a></div>
                        <div>🆘 <a href="https://github.com/caliari/Portafolio-IOL/issues" target="_blank" rel="noopener noreferrer">Centro de ayuda</a></div>
                    </div>
                    <div class='footer-links-card__disclaimer'>
                        Datos provistos sin garantía ni recomendación de inversión.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
