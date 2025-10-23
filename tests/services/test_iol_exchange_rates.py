"""Unit tests for the IOL exchange rates caching helpers."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import services.iol_exchange_rates as exchange_rates_module


@pytest.fixture
def exchange_rates_service(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Reload the module with a deterministic cache stub."""

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
    module = importlib.reload(exchange_rates_module)

    def advance(seconds: float) -> None:
        current_time["value"] += float(seconds)

    yield SimpleNamespace(
        module=module,
        advance=advance,
        ttl=lambda: ttl_holder["value"],
        cache_state=cache_state,
    )

    importlib.reload(exchange_rates_module)


def test_get_exchange_rates_parses_payload(exchange_rates_service: SimpleNamespace) -> None:
    module = exchange_rates_service.module
    payload = {
        "cotizacionCartera": "1.250,75",
        "cotizacionDolar": 950.5,
        "cuentas": [
            {
                "moneda": "Pesos",
                "disponible": "1000",
            }
        ],
    }
    client = SimpleNamespace(account_client=SimpleNamespace(fetch_account_status=Mock(return_value=payload)))

    result = module.get_exchange_rates(client)

    assert result == {
        "cotizacionCartera": pytest.approx(1250.75),
        "cotizacionDolar": pytest.approx(950.5),
    }


def test_get_exchange_rates_uses_cache(exchange_rates_service: SimpleNamespace) -> None:
    module = exchange_rates_service.module
    account_client = SimpleNamespace(
        fetch_account_status=Mock(
            return_value={
                "cotizacionCartera": 1000,
                "cotizacionDolar": 950,
            }
        )
    )
    client = SimpleNamespace(account_client=account_client)

    first = module.get_exchange_rates(client)
    second = module.get_exchange_rates(client)

    assert first == second
    account_client.fetch_account_status.assert_called_once()


def test_get_exchange_rates_respects_ttl(exchange_rates_service: SimpleNamespace) -> None:
    module = exchange_rates_service.module
    account_client = SimpleNamespace(
        fetch_account_status=Mock(
            return_value={
                "cotizacionCartera": 1000,
                "cotizacionDolar": 950,
            }
        )
    )
    client = SimpleNamespace(account_client=account_client)

    module.get_exchange_rates(client)
    exchange_rates_service.advance(1799)
    module.get_exchange_rates(client)
    assert account_client.fetch_account_status.call_count == 1

    exchange_rates_service.advance(2)
    module.get_exchange_rates(client)
    assert account_client.fetch_account_status.call_count == 2


def test_get_exchange_rates_ttl_configuration(exchange_rates_service: SimpleNamespace) -> None:
    assert exchange_rates_service.ttl() == 1800


def test_get_exchange_rates_handles_non_mapping(exchange_rates_service: SimpleNamespace) -> None:
    module = exchange_rates_service.module
    account_client = SimpleNamespace(fetch_account_status=Mock(return_value=[1, 2, 3]))
    client = SimpleNamespace(account_client=account_client)

    result = module.get_exchange_rates(client)

    assert result == {"cotizacionCartera": None, "cotizacionDolar": None}
