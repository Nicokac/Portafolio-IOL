import streamlit as st
from shared.utils import _as_float_or_none
from ui.palette import get_active_palette


def render_header(rates=None):
    """Render the application header with contextual actions."""

    pal = get_active_palette()

    st.markdown(
        """
        <div style="display:flex; gap:0.8rem; align-items:flex-start; flex-wrap:wrap;">
            <span style="font-size:2.2rem; line-height:1;">📈</span>
            <div>
                <h1 style="margin:0; font-size:1.8rem;">IOL — Portafolio en vivo</h1>
                <p style="margin:0.4rem 0 0; color:#555;">
                    Acceso operativo en modo <strong>solo lectura</strong> para monitorear posiciones y cotizaciones.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if rates:
        render_fx_summary_in_header(rates, palette=pal)


def render_fx_summary_in_header(rates: dict, palette=None):
    """Render a summary of foreign exchange rates in the header.

    Parameters
    ----------
    rates : dict
        Diccionario con las cotizaciones a mostrar.
    """
    if not rates:
        return

    pal = palette or get_active_palette()

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
            style = (
                "display:flex; flex-direction:column; gap:0.2rem; "
                f"background-color:{pal.highlight_bg}; "
                f"color:{pal.highlight_text}; "
                "padding:0.8rem 1rem; border-radius:0.75rem; text-align:center;"
            )
            value_str = (
                f"$ {value:,.2f}"
                .replace(",", "_")
                .replace(".", ",")
                .replace("_", ".")
                if value
                else "–"
            )
            st.markdown(
                f"""
                <div style="{style}">
                    <span style="font-size:0.85rem; letter-spacing:0.02em; text-transform:uppercase; opacity:0.85;">{label}</span>
                    <span style="font-size:1.5rem; font-weight:700; line-height:1;">{value_str}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
