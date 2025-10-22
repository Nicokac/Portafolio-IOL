"""Tests for the profile caching service."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import services.profile_service as profile_service_module


@pytest.fixture
def profile_service(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Reload the profile service with a deterministic cache stub."""

    import streamlit as st

    current_time = {"value": 0.0}
    cache_state: dict[tuple[Any, ...], dict[str, Any]] = {}
    ttl_holder = {"value": None}

    def fake_cache_data(func=None, *, ttl: int = 0, **__: Any):
        ttl_holder["value"] = ttl

        def decorator(fn):
            def wrapper(*args: Any, **kwargs: Any):
                key = (
                    tuple(id(arg) for arg in args),
                    tuple(sorted(kwargs.items())),
                )
                entry = cache_state.get(key)
                if entry is not None and current_time["value"] - entry["timestamp"] < ttl:
                    return entry["value"]
                result = fn(*args, **kwargs)
                cache_state[key] = {"value": result, "timestamp": current_time["value"]}
                return result

            wrapper._cache_state = cache_state  # type: ignore[attr-defined]
            return wrapper

        if func is not None:
            return decorator(func)
        return decorator

    def clear_cache() -> None:
        cache_state.clear()

    fake_cache_data.clear = clear_cache  # type: ignore[attr-defined]

    monkeypatch.setattr(st, "cache_data", fake_cache_data, raising=False)
    module = importlib.reload(profile_service_module)

    def advance(seconds: float) -> None:
        current_time["value"] += float(seconds)

    yield SimpleNamespace(module=module, advance=advance, ttl=lambda: ttl_holder["value"], cache_state=cache_state)

    importlib.reload(profile_service_module)


def test_fetch_profile_caches_within_ttl(profile_service: SimpleNamespace) -> None:
    module = profile_service.module
    client = SimpleNamespace(get_profile=Mock(return_value={"nombre": "Juan"}))

    first = module.fetch_profile(client)
    second = module.fetch_profile(client)

    assert first == second
    assert client.get_profile.call_count == 1


def test_fetch_profile_cache_clearing_forces_refresh(profile_service: SimpleNamespace) -> None:
    module = profile_service.module
    client = SimpleNamespace(get_profile=Mock(return_value={"nombre": "Ana"}))

    module.fetch_profile(client)
    assert client.get_profile.call_count == 1

    module.st.cache_data.clear()
    module.fetch_profile(client)

    assert client.get_profile.call_count == 2


def test_fetch_profile_respects_ttl(profile_service: SimpleNamespace) -> None:
    module = profile_service.module
    client = SimpleNamespace(get_profile=Mock(return_value={"nombre": "Luis"}))

    module.fetch_profile(client)
    profile_service.advance(1799)
    module.fetch_profile(client)
    assert client.get_profile.call_count == 1

    profile_service.advance(2)
    module.fetch_profile(client)
    assert client.get_profile.call_count == 2


def test_fetch_profile_ttl_configuration(profile_service: SimpleNamespace) -> None:
    assert profile_service.ttl() == 1800


def test_fetch_profile_propagates_to_session_state(profile_service: SimpleNamespace) -> None:
    module = profile_service.module
    client = SimpleNamespace(get_profile=Mock(return_value={"nombre": "Carla"}))

    result = module.fetch_profile(client)
    module.st.session_state["iol_user_profile"] = result

    assert module.st.session_state["iol_user_profile"] == {"nombre": "Carla"}
