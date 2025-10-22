import streamlit as st

from shared.version import __version__


def render_security_info() -> None:
    """Render a block describing how credentials are protected."""
    st.markdown(
        f"""
        <div style="padding:1rem; border-radius:0.75rem; background-color:rgba(0,0,0,0.04);">
            <div style="display:flex; gap:0.6rem; align-items:center; margin-bottom:0.6rem;">
                <span style="font-size:1.4rem;">🔒</span>
                <div>
                    <div style="font-size:1rem; font-weight:700;">Seguridad de credenciales</div>
                    <div style="font-size:0.85rem; color:#555;">Observabilidad operativa · versión {__version__}</div>
                </div>
            </div>
            <ul style="margin:0; padding-left:1.2rem; display:grid; gap:0.4rem;">
                <li><strong>Fernet</strong> protege tokens en reposo y en tránsito.</li>
                <li>Seguros dentro de <strong>Streamlit Secrets</strong> sin exposición pública.</li>
                <li>Claves locales cifradas; nada se replica en la nube.</li>
                <li>Borrado inmediato de contraseñas de <code>session_state</code>.</li>
            </ul>
            <p style="margin:0.8rem 0 0; font-size:0.85rem; color:#555;">
                Tus credenciales sólo viven en tu entorno y permanecen auditables durante la sesión.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


__all__ = ["render_security_info"]
