import streamlit as st
from shared.utils import _as_float_or_none
from ui.palette import get_active_palette

def render_header(rates=None):
    st.title("📈 IOL — Portafolio en vivo (solo lectura)")
    st.caption("Autenticación sólo para **leer** portafolio / cotizaciones. No se envían órdenes.")
    st.caption(
        "Datos provistos sin garantía ni recomendación de inversión. "
        "[Documentación](https://github.com/caliari/Portafolio-IOL#readme) · "
        "[Ayuda](https://github.com/caliari/Portafolio-IOL/issues)"
    )

    if rates:
        render_fx_summary_in_header(rates)

def render_fx_summary_in_header(rates: dict):
    if not rates:
        return

    pal = get_active_palette()

    # Extraer las cotizaciones necesarias
    valores = {
        "💵 Oficial": _as_float_or_none(rates.get("oficial")),
        "📈 MEP": _as_float_or_none(rates.get("mep")),
        "🏦 CCL": _as_float_or_none(rates.get("ccl")),
        "🪙 Cripto": _as_float_or_none(rates.get("cripto")),
        "💳 Tarjeta": _as_float_or_none(rates.get("tarjeta")),
        "🧧 Blue": _as_float_or_none(rates.get("blue")),
    }

    cols = st.columns(len(valores))

    for col, (label, value) in zip(cols, valores.items()):
        with col:
            st.markdown(f"""
                <div style="background-color:{pal.highlight_bg}; color:{pal.highlight_text}; padding:0.6em 1em; border-radius:0.5em;">
                    <strong>{label}</strong><br>
                    <span style="font-size:1.2em;">{f"$ {value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".") if value else '–'}</span>
                </div>
            """, unsafe_allow_html=True)