"""Unit tests for the CEDEAR ratio caching helper."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

import services.iol_ratios_service as ratios_module


class DummyResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


@pytest.fixture
def ratios_service(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
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
    module = importlib.reload(ratios_module)

    def advance(seconds: float) -> None:
        current_time["value"] += float(seconds)

    yield SimpleNamespace(
        module=module,
        advance=advance,
        ttl=lambda: ttl_holder["value"],
        cache_state=cache_state,
    )

    importlib.reload(ratios_module)


def test_get_ceear_ratio_parses_payload(ratios_service: SimpleNamespace) -> None:
    module = ratios_service.module
    response = DummyResponse(
        {
            "ratioCEDEAR": "10",
            "moneda": "USD",
            "mercadoBase": "NYSE",
        }
    )
    client = SimpleNamespace(
        api_base="https://api.example.com/api/v2",
        _request=Mock(return_value=response),
    )

    result = module.get_ceear_ratio("AAPL", client)

    assert result == {
        "ratioCEDEAR": pytest.approx(10.0),
        "moneda": "USD",
        "mercadoBase": "NYSE",
    }
    client._request.assert_called_once()
    called_url = client._request.call_args.kwargs.get("url") or client._request.call_args.args[1]
    assert called_url.endswith("/Titulos/AAPL")


def test_get_ceear_ratio_uses_cache(ratios_service: SimpleNamespace) -> None:
    module = ratios_service.module
    client = SimpleNamespace(
        api_base="https://api.example.com/api/v2",
        _request=Mock(return_value=DummyResponse({"ratioCEDEAR": 5, "moneda": "USD"})),
    )

    first = module.get_ceear_ratio("TSLA", client)
    second = module.get_ceear_ratio("TSLA", client)

    assert first == second
    client._request.assert_called_once()


def test_get_ceear_ratio_respects_ttl(ratios_service: SimpleNamespace) -> None:
    module = ratios_service.module
    client = SimpleNamespace(
        api_base="https://api.example.com/api/v2",
        _request=Mock(return_value=DummyResponse({"ratioCEDEAR": 2})),
    )

    module.get_ceear_ratio("MSFT", client)
    ratios_service.advance(1799)
    module.get_ceear_ratio("MSFT", client)
    assert client._request.call_count == 1

    ratios_service.advance(2)
    module.get_ceear_ratio("MSFT", client)
    assert client._request.call_count == 2


def test_get_ceear_ratio_ttl_configuration(ratios_service: SimpleNamespace) -> None:
    assert ratios_service.ttl() == 1800


def test_get_ceear_ratio_handles_invalid_payload(ratios_service: SimpleNamespace) -> None:
    module = ratios_service.module
    client = SimpleNamespace(
        api_base="https://api.example.com/api/v2",
        _request=Mock(return_value=DummyResponse([1, 2, 3])),
    )

    result = module.get_ceear_ratio("GGAL", client)

    assert result == {"ratioCEDEAR": None, "moneda": None, "mercadoBase": None}


def test_get_ceear_ratio_requires_symbol(ratios_service: SimpleNamespace) -> None:
    module = ratios_service.module
    client = SimpleNamespace(api_base="https://api.example.com/api/v2", _request=Mock())

    with pytest.raises(ValueError):
        module.get_ceear_ratio("", client)
