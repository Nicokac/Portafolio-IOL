"""Streamlit-bound helpers extracted from the legacy cache module."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Tuple
from uuid import uuid4

import streamlit as st

from infrastructure.iol.auth import IOLAuth, InvalidCredentialsError
from infrastructure.iol.client import IIOLProvider, build_iol_client as _build_iol_client
from services.health import record_fx_cache_usage, record_iol_refresh
from shared.cache import cache
from shared.settings import cache_ttl_fx, settings


logger = logging.getLogger(__name__)


def _trigger_logout() -> None:
    """Clear session and tokens triggering a fresh login."""

    try:
        from application import auth_service

        user = st.session_state.get("IOL_USERNAME", "")
        auth_service.logout(user)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("auto logout failed: %s", exc)
        raise


@cache.cache_resource
def get_client_cached(
    cache_key: str, user: str, tokens_file: Path | str | None
) -> IIOLProvider:
    auth = IOLAuth(
        user,
        "",
        tokens_file=tokens_file,
        allow_plain_tokens=settings.allow_plain_tokens,
    )
    try:
        auth.refresh()
        record_iol_refresh(True)
    except InvalidCredentialsError as err:
        auth.clear_tokens()
        st.session_state["force_login"] = True
        record_iol_refresh(False, detail="Credenciales inválidas")
        raise err
    except Exception as err:
        record_iol_refresh(False, detail=err)
        raise
    return _build_iol_client(user, "", tokens_file=tokens_file, auth=auth)


def get_fx_rates_cached():
    """Return FX rates cached in Streamlit session state."""

    now = time.time()
    ttl = cache_ttl_fx
    last = st.session_state.get("fx_rates_ts", 0)
    if "fx_rates" not in st.session_state or now - last > ttl:
        from services import cache as cache_module

        data, error = cache_module.fetch_fx_rates()
        st.session_state["fx_rates"] = data
        st.session_state["fx_rates_error"] = error
        st.session_state["fx_rates_ts"] = now
        record_fx_cache_usage("refresh", age=0.0)
    else:
        age = now - last if last else None
        record_fx_cache_usage("hit", age=age)
    return (
        st.session_state.get("fx_rates", {}),
        st.session_state.get("fx_rates_error"),
    )


def build_iol_client(
    user: str | None = None,
) -> Tuple[IIOLProvider | None, Exception | None]:
    user = user or st.session_state.get("IOL_USERNAME") or settings.IOL_USERNAME
    if not user:
        return None, RuntimeError("missing user")
    if "client_salt" not in st.session_state:
        st.session_state["client_salt"] = uuid4().hex
    salt = str(st.session_state.get("client_salt", ""))
    tokens_file = cache.get("tokens_file")
    if not tokens_file:
        sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", user)
        user_hash = hashlib.sha256(user.encode()).hexdigest()[:12]
        tokens_file = Path("tokens") / f"{sanitized}-{user_hash}.json"
        cache.set("tokens_file", str(tokens_file))
    cache_key = hashlib.sha256(
        f"{tokens_file}:{salt}".encode()
    ).hexdigest()
    st.session_state["cache_key"] = cache_key
    try:
        cli = get_client_cached(cache_key, user, tokens_file)
        return cli, None
    except InvalidCredentialsError as err:
        _trigger_logout()
        return None, err
    except Exception as err:
        logger.exception("build_iol_client failed: %s", err)
        return None, err


__all__ = [
    "_trigger_logout",
    "get_client_cached",
    "get_fx_rates_cached",
    "build_iol_client",
]
