"""Helpers to cache CEDEAR ratio metadata from the IOL security endpoint."""

from __future__ import annotations

from typing import Any, Mapping

import streamlit as st

from infrastructure.iol.client import IOLClient
from shared.utils import _to_float

CACHE_TTL_SECONDS = 1800


def _parse_ratio(value: Any) -> float | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return float(parsed)


def _normalize_payload(payload: Mapping[str, Any]) -> dict[str, float | str | None]:
    return {
        "ratioCEDEAR": _parse_ratio(payload.get("ratioCEDEAR")),
        "moneda": payload.get("moneda") if isinstance(payload.get("moneda"), str) else None,
        "mercadoBase": payload.get("mercadoBase") if isinstance(payload.get("mercadoBase"), str) else None,
    }


def _build_url(iol_client: IOLClient, symbol: str) -> str:
    base_url = getattr(iol_client, "_base", iol_client.api_base).rstrip("/")
    resolved_symbol = (symbol or "").strip().upper()
    if not resolved_symbol:
        raise ValueError("Symbol is required to fetch CEDEAR ratio metadata")
    return f"{base_url}/Titulos/{resolved_symbol}"


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def get_ceear_ratio(symbol: str, iol_client: IOLClient) -> dict[str, float | str | None]:
    """Return cached CEDEAR ratio metadata from ``/api/v2/Titulos/{simbolo}``."""

    url = _build_url(iol_client, symbol)
    response = iol_client._request("GET", url)
    if response is None:
        return {"ratioCEDEAR": None, "moneda": None, "mercadoBase": None}
    try:
        payload = response.json() or {}
    except ValueError:
        return {"ratioCEDEAR": None, "moneda": None, "mercadoBase": None}
    if not isinstance(payload, Mapping):
        return {"ratioCEDEAR": None, "moneda": None, "mercadoBase": None}
    return _normalize_payload(payload)


__all__ = ["get_ceear_ratio"]
