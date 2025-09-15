"""Network error propagation for service helpers."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services import cache as svc_cache
from shared.errors import NetworkError


@pytest.fixture(autouse=True)
def streamlit_state(monkeypatch: pytest.MonkeyPatch):
    """Provide an isolated Streamlit session state for cache decorators."""
    state: dict = {}
    monkeypatch.setattr(svc_cache, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr("shared.cache.st", SimpleNamespace(session_state=state))
    svc_cache.get_client_cached.clear()
    svc_cache.fetch_fx_rates.clear()
    yield
    svc_cache.get_client_cached.clear()
    svc_cache.fetch_fx_rates.clear()


def test_get_client_cached_raises_network_error(monkeypatch: pytest.MonkeyPatch):
    """Refreshing tokens should surface IOL network errors to callers."""

    class DummyAuth:
        def __init__(self, *args, **kwargs):
            pass

        def refresh(self):
            raise NetworkError("offline")

        def clear_tokens(self):  # pragma: no cover - defensive
            pass

    monkeypatch.setattr(svc_cache, "IOLAuth", DummyAuth)
    recorder = MagicMock()
    monkeypatch.setattr(svc_cache, "record_iol_refresh", recorder)

    with pytest.raises(NetworkError):
        svc_cache.get_client_cached("cache-key", "user", None)

    recorder.assert_called_once()
    args, kwargs = recorder.call_args
    assert args[0] is False
    assert isinstance(kwargs.get("detail"), NetworkError)


def test_fetch_fx_rates_propagates_timeout(monkeypatch: pytest.MonkeyPatch):
    """Timeouts from the FX provider should not be swallowed by the cache layer."""
    monkeypatch.setattr(svc_cache, "record_fx_api_response", MagicMock())

    class DummyProvider:
        def get_rates(self):
            raise TimeoutError("too slow")

    monkeypatch.setattr(svc_cache, "get_fx_provider", lambda: DummyProvider())

    with pytest.raises(TimeoutError):
        svc_cache.fetch_fx_rates()

