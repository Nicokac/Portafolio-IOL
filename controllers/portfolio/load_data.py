import logging

import pandas as pd
import streamlit as st

from services.cache import fetch_portfolio
from services.performance_timer import performance_timer
from shared.errors import AppError

logger = logging.getLogger(__name__)


def load_portfolio_data(cli, psvc):
    """Fetch and normalize portfolio positions."""
    tokens_path = getattr(getattr(cli, "auth", None), "tokens_path", None)
    payload = None
    with st.spinner("Cargando y actualizando portafolio... ⏳"):
        telemetry: dict[str, object] = {"status": "success", "source": "api"}
        try:
            with performance_timer("portfolio_load_data", extra=telemetry):
                try:
                    payload = fetch_portfolio(cli)
                except AppError as err:
                    telemetry["status"] = "error"
                    telemetry["detail"] = err.__class__.__name__
                    raise
                except Exception:
                    telemetry["status"] = "error"
                    telemetry["detail"] = "exception"
                    raise
                if isinstance(payload, dict) and payload.get("_cached"):
                    telemetry["source"] = "cache"
        except AppError as err:
            st.error(str(err))
            st.stop()
        except Exception:  # pragma: no cover - streamlit error path
            logger.exception(
                "Error al consultar portafolio",
                extra={"tokens_file": tokens_path},
            )
            st.error("No se pudo cargar el portafolio, intente más tarde")
            st.stop()

    auth_error = False
    if isinstance(payload, dict):
        msg = str(payload.get("message", ""))
        auth_error = (
            payload.get("status") in (401, 403)
            or payload.get("code") in (401, 403)
            or "unauthorized" in msg.lower()
            or "no autorizado" in msg.lower()
        )

    if st.session_state.get("force_login") or auth_error:
        st.warning("Sesión expirada, por favor vuelva a iniciar sesión")
        st.rerun()
    elif isinstance(payload, dict) and payload.get("_cached"):
        st.warning(
            "No se pudo contactar a IOL; mostrando datos del portafolio en caché."
        )

    if isinstance(payload, dict) and "message" in payload:
        st.info(f"ℹ️ Mensaje de IOL: \"{payload['message']}\"")
        st.stop()

    df_pos = psvc.normalize_positions(payload)
    if df_pos.empty:
        logger.info(
            "Portafolio vacío pero API respondió correctamente",
            extra={"tokens_file": tokens_path},
        )
        st.warning(
            "No se encontraron posiciones o no pudimos mapear la respuesta."
        )
        if isinstance(payload, dict) and "activos" in payload:
            st.dataframe(pd.DataFrame(payload["activos"]).head(20))
            st.caption("Ejemplo de datos recibidos del portafolio")
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

