import logging

import pandas as pd
import streamlit as st

from services.cache import fetch_portfolio
from services.performance_timer import performance_timer
from shared.errors import AppError

logger = logging.getLogger(__name__)


def _filters_active() -> bool:
    state = getattr(st, "session_state", {})
    symbol_query = str(state.get("symbol_query", "") or "").strip()
    if symbol_query:
        return True

    selected_syms = state.get("selected_syms")
    last_all = state.get("portfolio_last_all_symbols")
    if isinstance(selected_syms, list):
        cleaned = [str(sym).strip() for sym in selected_syms if str(sym).strip()]
        if not cleaned and (last_all or symbol_query):
            return True
        if last_all and isinstance(last_all, list) and 0 < len(cleaned) < len(last_all):
            return True

    selected_types = state.get("selected_types")
    last_types = state.get("portfolio_last_available_types")
    if isinstance(selected_types, list):
        cleaned_types = [str(t).strip() for t in selected_types if str(t).strip()]
        if not cleaned_types and last_types:
            return True
        if (
            isinstance(last_types, list)
            and last_types
            and 0 < len(cleaned_types) < len(last_types)
        ):
            return True

    return False


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
        st.session_state["force_login"] = True
        st.error("No se pudo autenticar con IOL")
        st.stop()
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
        if _filters_active():
            st.info("No se encontraron activos que cumplan los filtros.")
        else:
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
    try:
        st.session_state["portfolio_last_all_symbols"] = all_symbols
        st.session_state["portfolio_last_available_types"] = available_types
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudieron almacenar los metadatos de filtros", exc_info=True)
    return df_pos, all_symbols, available_types

