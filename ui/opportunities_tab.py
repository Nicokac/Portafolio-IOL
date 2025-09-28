from __future__ import annotations

import streamlit as st

from controllers.opportunities import search_opportunities
from controllers.opportunities_spec import OPPORTUNITIES_SPEC
from shared.version import __version__


def _format_header() -> str:
    return f"🚀 Oportunidades de mercado · versión {__version__}"


def render_opportunities_tab() -> None:
    """Render the opportunities exploration tab."""

    st.header(_format_header())
    st.caption(
        "Explora ideas de inversión basadas en métricas cuantitativas y volumen de mercado."
    )

    if st.button("Buscar oportunidades", type="primary"):
        df = search_opportunities()
        if df is None or df.empty:
            st.info("No se encontraron oportunidades disponibles.")
            return

        missing = [col for col in OPPORTUNITIES_SPEC.columns if col not in df.columns]
        if missing:
            st.error(
                "Faltan columnas esperadas en el resultado: " + ", ".join(missing)
            )
            return

        st.dataframe(df[OPPORTUNITIES_SPEC.columns], width="stretch")
        st.caption(
            "Los montos y volúmenes se expresan en sus monedas originales."
        )
    else:
        st.caption("Usá el botón para obtener las oportunidades más recientes.")


__all__ = ["render_opportunities_tab"]
