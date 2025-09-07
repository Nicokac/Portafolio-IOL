import streamlit as st
from ui.header import render_header


def render_footer() -> None:
    """Render developer information at the bottom of the login form."""
    st.caption("Desarrollado por Nicolás Kachuk")


def render_login_page() -> None:
    """Display the login form with header and footer."""
    render_header()
    with st.form("login_form"):
        st.text_input("Usuario", key="IOL_USERNAME")
        st.text_input("Contraseña", type="password", key="IOL_PASSWORD")
        st.form_submit_button("Iniciar sesión")
        render_footer()
