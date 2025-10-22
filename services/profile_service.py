"""Caching helpers for retrieving the authenticated investor profile."""

from __future__ import annotations

from typing import Any

import streamlit as st

from infrastructure.iol.client import IOLClient


@st.cache_data(ttl=1800)
def fetch_profile(iol_client: IOLClient) -> dict[str, Any] | None:
    """Return the cached profile payload for the authenticated client."""

    return iol_client.get_profile()


__all__ = ["fetch_profile"]
