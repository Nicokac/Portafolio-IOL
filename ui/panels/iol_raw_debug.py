"""Panel de diagnóstico para capturar y comparar payloads RAW de IOL."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st

from application.portfolio_service import capture_iol_raw_snapshot, to_iol_format
from shared.redact import redact_secrets
from shared.time_provider import TimeProvider

logger = logging.getLogger(__name__)

_SYMBOL = "BPOC7"
_MARKET = "bcba"
_COUNTRY = "argentina"
_NOTE = "Los datos aquí mostrados son crudos desde la API de IOL (sin reescalado)."


def _get_positions_dataframe() -> pd.DataFrame:
    state = getattr(st, "session_state", {})
    viewmodel = state.get("portfolio_last_viewmodel")
    if viewmodel is not None:
        positions = getattr(viewmodel, "positions", None)
        if isinstance(positions, pd.DataFrame):
            return positions
    dataset = state.get("portfolio_last_positions")
    if isinstance(dataset, pd.DataFrame):
        return dataset
    return pd.DataFrame()


def _safe_filename(timestamp: str | None) -> str:
    if timestamp:
        safe = timestamp.replace(":", "").replace("/", "-").replace(" ", "T")
        return f"iol_raw_{_SYMBOL}_{safe}.json"
    fallback = TimeProvider.now_datetime().strftime("%Y%m%dT%H%M%S")
    return f"iol_raw_{_SYMBOL}_{fallback}.json"


def _prepare_download_payload(snapshot: Mapping[str, Any]) -> tuple[bytes, str]:
    sanitized = redact_secrets(snapshot)
    payload = json.dumps(sanitized, ensure_ascii=False, indent=2)
    file_name = _safe_filename(str(snapshot.get("ts")))
    return payload.encode("utf-8-sig"), file_name


def render_iol_raw_debug_panel() -> None:
    """Renderiza el panel de auditoría RAW de IOL."""

    st.header("🔍 IOL RAW")
    st.caption(_NOTE)

    state = getattr(st, "session_state", {})
    last_snapshot = state.get("iol_raw_last_snapshot")

    if st.button("🔍 Capturar IOL RAW (BPOC7)"):
        cli = state.get("cli")
        if cli is None:
            st.warning("No hay un cliente IOL autenticado en la sesión actual.")
        else:
            try:
                with st.spinner("Consultando API de IOL..."):
                    snapshot = capture_iol_raw_snapshot(
                        cli,
                        symbol=_SYMBOL,
                        mercado=_MARKET,
                        country=_COUNTRY,
                    )
            except Exception:  # pragma: no cover - defensive guard
                st.error("No se pudo capturar el snapshot RAW de IOL.")
                logger.debug("Captura RAW de IOL fallida", exc_info=True)
            else:
                state["iol_raw_last_snapshot"] = snapshot
                last_snapshot = snapshot

    if not isinstance(last_snapshot, Mapping):
        st.info(
            "Aún no se capturó ningún snapshot RAW. Utilizá el botón para obtener uno reciente.",
        )
        return

    sanitized_snapshot = redact_secrets(last_snapshot)
    portfolio_row = sanitized_snapshot.get("portfolio_row")
    portfolio_raw = sanitized_snapshot.get("portfolio_raw")
    quote_raw = sanitized_snapshot.get("quote_raw")
    quote_detail_raw = sanitized_snapshot.get("quote_detail_raw")

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("📁 Portafolio RAW")
        if isinstance(portfolio_row, Mapping):
            st.json(portfolio_row)
        else:
            st.warning(f"{_SYMBOL} no figura en el portafolio más reciente.")
        with st.expander("Ver portafolio completo"):
            st.json(portfolio_raw)

    with right_col:
        st.subheader("💹 Cotizaciones RAW")
        st.json(quote_raw)
        st.json(quote_detail_raw)

    download_bytes, file_name = _prepare_download_payload(last_snapshot)
    st.download_button(
        "💾 Descargar snapshot RAW",
        data=download_bytes,
        file_name=file_name,
        mime="application/json",
    )

    df_positions = _get_positions_dataframe()
    if not df_positions.empty:
        df_formatted = to_iol_format(df_positions)
        activo_series = df_formatted.get("Activo")
        if activo_series is not None:
            mask = activo_series.astype(str).str.upper() == _SYMBOL
        else:
            mask = pd.Series([], dtype=bool)
        if getattr(mask, "any", lambda: False)():
            st.subheader("📊 Comparativa formateada (vista actual)")
            st.dataframe(df_formatted.loc[mask], hide_index=True)


__all__ = ["render_iol_raw_debug_panel"]
