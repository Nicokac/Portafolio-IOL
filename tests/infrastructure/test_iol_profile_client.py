"""Tests for the investor profile endpoint integration."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import requests

from infrastructure.iol import client as iol_client_module


pytestmark = pytest.mark.usefixtures("restore_real_iol_client")


class _DummyAuth:
    """Authentication stub tracking token usage."""

    def __init__(self) -> None:
        self.ensure_calls = 0
        self.refresh_calls = 0
        self.header_calls = 0
        self.tokens = {"access_token": "dummy"}
        self.tokens_path = "tokens.json"

    def ensure_token(self, *, silent: bool = False) -> None:
        self.ensure_calls += 1

    def auth_header(self) -> dict[str, str]:
        self.header_calls += 1
        return {"Authorization": "Bearer dummy"}

    def refresh(self) -> None:
        self.refresh_calls += 1


@pytest.fixture
def client_factory(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return a factory that builds an ``IOLClient`` with safe defaults."""

    monkeypatch.setattr(iol_client_module.LegacySession, "get", classmethod(lambda cls: None))
    monkeypatch.setattr(iol_client_module.IOLClient, "_ensure_market_auth", lambda self: None)

    def _factory(auth: _DummyAuth | None = None) -> iol_client_module.IOLClient:
        return iol_client_module.IOLClient("user", "pass", auth=auth or _DummyAuth())

    return _factory


def _make_response(payload: Any) -> SimpleNamespace:
    return SimpleNamespace(status_code=200, content=b"{}", json=lambda: payload, raise_for_status=lambda: None)


def test_get_profile_returns_normalized_payload(monkeypatch: pytest.MonkeyPatch, client_factory: Any) -> None:
    tz = ZoneInfo("UTC")
    expiry = datetime(2024, 6, 1, 12, 30, tzinfo=tz)

    payload = {
        "nombre": "  Juan Pérez  ",
        "testInversor": {
            "perfil": "Moderado",
            "vigente": True,
            "fechaVencimiento": expiry.isoformat(),
        },
    }

    def fake_request(self: iol_client_module.IOLClient, method: str, url: str, **_: Any) -> SimpleNamespace:
        assert method == "GET"
        assert url == iol_client_module.PROFILE_URL
        return _make_response(payload)

    reference = datetime(2024, 5, 1, 12, 0, tzinfo=iol_client_module.TimeProvider.timezone())
    monkeypatch.setattr(
        iol_client_module.TimeProvider,
        "now_datetime",
        classmethod(lambda cls: reference),
    )

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", fake_request)
    client = client_factory()

    result = client.get_profile()

    assert result is not None
    assert result["nombre"] == "Juan Pérez"
    assert result["perfil_inversor"] == "Moderado"
    assert result["preferencias"] == ["CEDEARs Diversificados", "FCI Balanceados"]
    assert isinstance(result["vigencia"], datetime)
    expected_vigencia = expiry.astimezone(iol_client_module.TimeProvider.timezone())
    assert result["vigencia"].isoformat() == expected_vigencia.isoformat()


def test_get_profile_handles_missing_fields(monkeypatch: pytest.MonkeyPatch, client_factory: Any) -> None:
    payload = {"nombre": None, "testInversor": {}}

    monkeypatch.setattr(
        iol_client_module.IOLClient,
        "_request",
        lambda self, method, url: _make_response(payload),
    )
    client = client_factory()

    result = client.get_profile()

    assert result == {
        "nombre": None,
        "perfil_inversor": None,
        "vigencia": None,
        "preferencias": None,
    }


@pytest.mark.parametrize(
    "vigente,offset,expected",
    [
        (False, timedelta(hours=1), None),
        (True, timedelta(hours=-1), None),
    ],
)
def test_get_profile_returns_none_when_expired(
    monkeypatch: pytest.MonkeyPatch,
    client_factory: Any,
    vigente: bool,
    offset: timedelta,
    expected: Any,
) -> None:
    tz = iol_client_module.TimeProvider.timezone()
    now = datetime(2024, 5, 10, 9, 0, tzinfo=tz)

    def fixed_now(cls: type[iol_client_module.TimeProvider]) -> datetime:
        return now

    monkeypatch.setattr(
        iol_client_module.TimeProvider,
        "now_datetime",
        classmethod(fixed_now),
    )

    expiry = (now + offset).astimezone(ZoneInfo("UTC"))
    payload = {
        "nombre": "Juan",
        "testInversor": {
            "perfil": "Moderado",
            "vigente": vigente,
            "fechaVencimiento": expiry.isoformat(),
        },
    }

    monkeypatch.setattr(
        iol_client_module.IOLClient,
        "_request",
        lambda self, method, url: _make_response(payload),
    )
    client = client_factory()

    assert client.get_profile() is expected


def test_get_profile_refreshes_token_on_401(monkeypatch: pytest.MonkeyPatch, client_factory: Any) -> None:
    auth = _DummyAuth()
    tz = ZoneInfo("UTC")
    expiry = datetime(2024, 6, 15, 15, 0, tzinfo=tz)
    payload = {
        "nombre": "Ana",
        "testInversor": {
            "perfil": "Conservador",
            "vigente": True,
            "fechaVencimiento": expiry.isoformat(),
        },
    }

    class DummyResponse:
        def __init__(self, status_code: int, payload: Any | None = None) -> None:
            self.status_code = status_code
            self._payload = payload
            self.content = b"{}" if payload is not None else b""

        def json(self) -> Any:
            return self._payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.HTTPError(response=self)

    responses = [DummyResponse(401), DummyResponse(200, payload)]
    calls: list[dict[str, Any]] = []

    reference = expiry.astimezone(iol_client_module.TimeProvider.timezone()) - timedelta(days=1)
    monkeypatch.setattr(
        iol_client_module.TimeProvider,
        "now_datetime",
        classmethod(lambda cls: reference),
    )

    def fake_session_request(self, method: str, url: str, **kwargs: Any) -> DummyResponse:
        calls.append({"method": method, "url": url, "headers": dict(kwargs.get("headers", {}))})
        return responses.pop(0)

    monkeypatch.setattr(requests.Session, "request", fake_session_request)
    client = client_factory(auth=auth)

    result = client.get_profile()

    assert result is not None
    assert auth.refresh_calls == 1
    assert len(calls) == 2
    assert all(call["method"] == "GET" and call["url"] == iol_client_module.PROFILE_URL for call in calls)
    assert result["perfil_inversor"] == "Conservador"


@pytest.mark.parametrize("status_code", [404, 500])
def test_get_profile_handles_http_errors(
    monkeypatch: pytest.MonkeyPatch,
    client_factory: Any,
    caplog: pytest.LogCaptureFixture,
    status_code: int,
) -> None:
    caplog.set_level("WARNING", logger=iol_client_module.logger.name)

    if status_code == 404:
        monkeypatch.setattr(
            iol_client_module.IOLClient,
            "_request",
            lambda self, method, url: None,
        )
    else:
        error_response = SimpleNamespace(status_code=status_code)

        def failing_request(self: iol_client_module.IOLClient, method: str, url: str) -> None:
            raise requests.HTTPError("server error", response=error_response)

        monkeypatch.setattr(iol_client_module.IOLClient, "_request", failing_request)

    client = client_factory()

    assert client.get_profile() is None
    if status_code == 500:
        assert any("get_profile request error" in record.message for record in caplog.records)
