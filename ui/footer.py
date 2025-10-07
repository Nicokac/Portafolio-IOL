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
                justify-content: space-between;
                gap: 1.5rem;
                font-size: 0.9rem;
                color: #343a40;
                margin-top: 0.75rem;
            }}
            .footer-column {{
                flex: 1 1 240px;
                min-width: 200px;
            }}
            .footer-title {{
                font-weight: 700;
                font-size: 0.95rem;
                margin-bottom: 0.5rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #212529;
            }}
            .footer-links a {{
                color: #4f6f8f;
                text-decoration: underline;
                font-weight: 600;
            }}
            .footer-links a:hover,
            .footer-links a:focus {{
                color: #3c4f65;
            }}
            .footer-meta {{
                color: #5c636a;
            }}
            .footer-meta strong {{
                color: #495057;
                font-weight: 600;
            }}
            .footer-disclaimer {{
                font-size: 0.75rem;
                color: #6c757d;
                margin-top: 0.75rem;
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
                <div class='footer-title'>Enlaces útiles</div>
                <div class='footer-links'>
                    <p><a href='https://github.com/caliari/Portafolio-IOL#readme' target='_blank' rel='noopener noreferrer'>README y Wiki</a></p>
                    <p><a href='https://github.com/caliari/Portafolio-IOL/wiki/Troubleshooting' target='_blank' rel='noopener noreferrer'>Guía de troubleshooting</a></p>
                    <p><a href='https://github.com/caliari/Portafolio-IOL/issues' target='_blank' rel='noopener noreferrer'>Centro de ayuda y soporte</a></p>
                </div>
            </div>
            <div class='footer-column'>
                <div class='footer-title'>Información operativa</div>
                <div class='footer-meta'>
                    <p><strong>Versión:</strong> {version}</p>
                    <p><strong>Última sincronización:</strong> {timestamp}</p>
                    <p>Desarrollado por Nicolás K. · <a href='https://github.com/caliari' target='_blank' rel='noopener noreferrer'>Portafolio</a></p>
                    <div class='footer-disclaimer'>
                        &copy; {year}. Los datos se ofrecen sin garantía. Uso bajo su responsabilidad.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
