"""Helpers to cache exchange rate information from the IOL account status."""

from __future__ import annotations

from typing import Any, Mapping

import streamlit as st

from infrastructure.iol.client import IOLClient
from shared.utils import _to_float

CACHE_TTL_SECONDS = 1800


def _parse_rate(value: Any) -> float | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return float(parsed)


def _normalize_payload(payload: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        "cotizacionCartera": _parse_rate(payload.get("cotizacionCartera")),
        "cotizacionDolar": _parse_rate(payload.get("cotizacionDolar")),
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_exchange_rates(iol_client: IOLClient) -> dict[str, float | None]:
    """Return cached exchange rates obtained from ``/api/v2/estadocuenta``."""

    payload = iol_client.account_client.fetch_account_status()
    if not isinstance(payload, Mapping):
        return {"cotizacionCartera": None, "cotizacionDolar": None}
    return _normalize_payload(payload)


__all__ = ["get_exchange_rates"]
