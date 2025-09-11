import pandas as pd
import streamlit as st

from services.cache import fetch_portfolio


def load_portfolio_data(cli, psvc):
    """Fetch and normalize portfolio positions."""
    payload = None
    with st.spinner("Cargando y actualizando portafolio... ⏳"):
        try:
            payload = fetch_portfolio(cli)
        except Exception as e:  # pragma: no cover - streamlit error path
            st.error(f"Error al consultar portafolio: {e}")
            st.stop()

    if isinstance(payload, dict) and payload.get("_cached"):
        st.warning(
            "No se pudo contactar a IOL; mostrando datos del portafolio en caché."
        )

    if isinstance(payload, dict) and "message" in payload:
        st.info(f"ℹ️ Mensaje de IOL: \"{payload['message']}\"")
        st.stop()

    df_pos = psvc.normalize_positions(payload)
    if df_pos.empty:
        st.warning(
            "No se encontraron posiciones o no pudimos mapear la respuesta."
        )
        if isinstance(payload, dict) and "activos" in payload:
            st.dataframe(pd.DataFrame(payload["activos"]).head(20))
        st.stop()

    all_symbols = sorted(df_pos["simbolo"].astype(str).str.upper().unique())
    available_types = sorted(
        {
            psvc.classify_asset_cached(s)
            for s in all_symbols
            if psvc.classify_asset_cached(s)
        }
    )
    return df_pos, all_symbols, available_types
