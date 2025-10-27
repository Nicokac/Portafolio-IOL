"""Portfolio caching helpers extracted from the legacy cache module."""

from __future__ import annotations

import logging
import time

import requests

try:  # pragma: no cover - optional dependency in tests
    import streamlit as st
except Exception:  # pragma: no cover - Streamlit not available during some tests
    st = None  # type: ignore

try:  # pragma: no cover - bootstrap module optional in some tests
    from bootstrap.config import TOTAL_LOAD_START
except Exception:  # pragma: no cover - fallback when bootstrap not initialized
    TOTAL_LOAD_START = None  # type: ignore[assignment]

from infrastructure.iol.auth import InvalidCredentialsError
from infrastructure.iol.client import IIOLProvider
from services.cache.quotes import _resolve_auth_ref
from services.cache.ui_adapter import _trigger_logout
from services.health import record_portfolio_load
from shared.cache import cache
from shared.errors import NetworkError
from shared.errors import TimeoutError as SharedTimeoutError
from shared.settings import cache_ttl_portfolio
from shared.telemetry import log_metric

__all__ = ["fetch_portfolio"]

logger = logging.getLogger(__name__)

_BOOTSTRAP_METRIC_FLAG = "_rerun_bootstrap_logged"
_FALLBACK_BOOTSTRAP_LOGGED = False


def _record_bootstrap_metric() -> None:
    global _FALLBACK_BOOTSTRAP_LOGGED
    if TOTAL_LOAD_START is None:
        return
    try:
        elapsed_ms = max((time.perf_counter() - float(TOTAL_LOAD_START)) * 1000.0, 0.0)
    except Exception:
        return

    state = None
    if st is not None:
        try:
            state = st.session_state
        except Exception:  # pragma: no cover - defensive safeguard
            state = None

    if state is not None:
        try:
            if state.get(_BOOTSTRAP_METRIC_FLAG):
                return
        except Exception:
            logger.debug("Unable to check bootstrap metric flag", exc_info=True)
            state = None

    else:
        if _FALLBACK_BOOTSTRAP_LOGGED:
            return

    try:
        log_metric(
            "rerun_bootstrap_ms",
            duration_ms=elapsed_ms,
            context={"phase": "portfolio_fetch"},
        )
    except Exception:  # pragma: no cover - telemetry best effort
        logger.debug("Unable to log rerun_bootstrap_ms", exc_info=True)

    if state is not None:
        try:
            state[_BOOTSTRAP_METRIC_FLAG] = True
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug("Unable to mark bootstrap metric in session_state", exc_info=True)
    else:
        _FALLBACK_BOOTSTRAP_LOGGED = True


@cache.cache_data(ttl=cache_ttl_portfolio)
def fetch_portfolio(_cli: IIOLProvider):
    start = time.time()
    _record_bootstrap_metric()
    tokens_path = getattr(getattr(_cli, "auth", None), "tokens_path", None)
    try:
        data = _cli.get_portfolio()
    except InvalidCredentialsError:
        auth = _resolve_auth_ref(_cli)
        if auth is not None:
            try:
                auth.clear_tokens()
            except Exception:
                pass
        _trigger_logout()
        logger.info(
            "fetch_portfolio using cache due to invalid credentials",
            extra={"tokens_file": tokens_path},
        )
        record_portfolio_load(None, source="cache", detail="invalid-credentials")
        return {"_cached": True}
    except requests.Timeout as exc:
        logger.info(
            "fetch_portfolio failed due to network timeout: %s",
            exc,
            extra={"tokens_file": tokens_path},
        )
        record_portfolio_load(None, source="error", detail="timeout")
        raise SharedTimeoutError("Error de red al consultar el portafolio") from exc
    except requests.RequestException as exc:
        logger.info(
            "fetch_portfolio failed due to network error: %s",
            exc,
            extra={"tokens_file": tokens_path},
        )
        record_portfolio_load(None, source="error", detail="network-error")
        raise NetworkError("Error de red al consultar el portafolio") from exc

    elapsed = (time.time() - start) * 1000
    record_portfolio_load(elapsed, source="api")
    log = logger.warning if elapsed > 600 else logger.info
    log(
        "fetch_portfolio done in %.0fms",
        elapsed,
        extra={"tokens_file": tokens_path},
    )
    return data
