import streamlit as st
from shared.version import __version__


def render_security_info() -> None:
    """Render a block describing how credentials are protected."""
    st.markdown(
        (
            "### 🔒 Seguridad de tus credenciales\n\n"
            "- Cifrado de tokens con [Fernet](https://cryptography.io/en/latest/fernet/)\n"
            "- Almacenamiento de secretos con [Streamlit Secrets](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/secrets-management)\n"
            "- Tokens guardados en archivos cifrados locales (no en la nube)\n"
            "- Limpieza inmediata de contraseñas en `session_state`\n\n"
            "Tus credenciales nunca se almacenan en servidores externos. El acceso a IOL se realiza de forma segura mediante tokens cifrados, protegidos con clave Fernet y gestionados localmente por la aplicación.\n\n"
            f"Estas medidas de seguridad aplican a la versión {__version__}."
        )
    )


__all__ = ["render_security_info"]
