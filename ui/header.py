# ui\header.py
import streamlit as st

def render_header():
    st.title("📈 IOL — Portafolio en vivo (solo lectura)")
    st.caption("Autenticación sólo para **leer** portafolio / cotizaciones. No se envían órdenes.")
    st.caption(
        "Datos provistos sin garantía ni recomendación de inversión. "
        "[Documentación](https://github.com/caliari/Portafolio-IOL#readme) · "
        "[Ayuda](https://github.com/caliari/Portafolio-IOL/issues)"
    )