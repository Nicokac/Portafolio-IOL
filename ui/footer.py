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
        <style>
            .footer-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 1.75rem;
                font-size: 0.9rem;
                color: #343a40;
                margin-top: 0.75rem;
            }}
            .footer-column {{
                flex: 1 1 260px;
                min-width: 220px;
            }}
            .footer-title {{
                font-weight: 700;
                font-size: 0.95rem;
                margin-bottom: 0.45rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #212529;
            }}
            .footer-meta {{
                color: #5c636a;
                line-height: 1.6;
            }}
            .footer-meta strong {{
                color: #495057;
                font-weight: 600;
            }}
            .footer-links-card-wrapper {{
                flex: 1 1 100%;
            }}
            .footer-links-card-wrapper a {{
                color: #4f6f8f;
                text-decoration: underline;
                font-weight: 600;
            }}
            .footer-links-card-wrapper a:hover,
            .footer-links-card-wrapper a:focus {{
                color: #3c4f65;
            }}
            @media (max-width: 576px) {{
                .footer-container {{
                    flex-direction: column;
                }}
            }}
        </style>
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
                    <p><strong>v0.3.4.3.1</strong> Sidebar vertical y header sin bloques redundantes.</p>
                    <p>Explorá el tab Monitoreo para revisar el healthcheck completo.</p>
                </div>
            </div>
            <div class='footer-links-card-wrapper'>
                <div style="padding: 0.8rem 1rem; border-radius: 0.6rem; background-color: rgba(0, 0, 0, 0.04); font-size: 0.95rem;">
                    <div style="font-weight: 600; margin-bottom: 0.4rem;">Enlaces útiles</div>
                    <div>📘 <a href="https://github.com/caliari/Portafolio-IOL#readme" target="_blank" rel="noopener noreferrer">Documentación</a></div>
                    <div>🆘 <a href="https://github.com/caliari/Portafolio-IOL/issues" target="_blank" rel="noopener noreferrer">Centro de ayuda</a></div>
                    <div style="margin-top: 0.6rem; color: rgb(102, 102, 102); font-size: 0.85rem;">
                        Datos provistos sin garantía ni recomendación de inversión.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
