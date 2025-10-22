"""Tests for the notification support in :mod:`infrastructure.iol.client`."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
import requests
from zoneinfo import ZoneInfo

from infrastructure.iol import client as iol_client_module
from tests.fixtures.auth import FakeAuth


class _StreamlitStub:
    """Minimal Streamlit stub exposing a mutable ``session_state``."""

    session_state: dict[str, Any] = {}


class _StubIol:
    """Avoid hitting the real IOL SDK during client initialisation."""

    def __init__(self, user: str, password: str) -> None:
        self.user = user
        self.password = password
        self.gestionar_calls = 0

    def gestionar(self) -> None:
        self.gestionar_calls += 1


@pytest.fixture(autouse=True)
def _patch_streamlit(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Ensure every test operates with a clean Streamlit session."""

    _StreamlitStub.session_state = {}
    monkeypatch.setattr(iol_client_module, "st", _StreamlitStub)
    monkeypatch.setattr(iol_client_module, "Iol", _StubIol)
    return _StreamlitStub.session_state


@pytest.fixture
def fixed_now(monkeypatch: pytest.MonkeyPatch) -> datetime:
    tz = iol_client_module.TimeProvider.timezone()
    now = datetime(2024, 4, 20, 12, 0, tzinfo=tz)

    def _now_datetime(cls: type[iol_client_module.TimeProvider]) -> datetime:
        return now

    monkeypatch.setattr(
        iol_client_module.TimeProvider,
        "now_datetime",
        classmethod(_now_datetime),
    )
    return now


def _response(payload: Any, *, status_code: int = 200) -> SimpleNamespace:
    return SimpleNamespace(
        status_code=status_code,
        content=b"{}" if payload is not None else b"",
        json=lambda: payload,
    )


def _build_client(monkeypatch: pytest.MonkeyPatch) -> iol_client_module.IOLClient:
    # Skip expensive legacy session bootstrapping.
    monkeypatch.setattr(iol_client_module.LegacySession, "get", classmethod(lambda cls: None))
    return iol_client_module.IOLClient("user", "pass", auth=FakeAuth())


def test_get_notification_returns_valid_payload(monkeypatch: pytest.MonkeyPatch, fixed_now: datetime) -> None:
    tz = ZoneInfo("UTC")
    future = datetime(2024, 4, 20, 15, 0, tzinfo=tz)

    def fake_request(self: iol_client_module.IOLClient, method: str, url: str, **_: Any) -> SimpleNamespace:
        assert method == "GET"
        assert url.endswith("/Notificacion")
        return _response(
            {
                "mensaje": "Mantenimiento programado",
                "activo": True,
                "vigencia": future.isoformat(),
                "vigenciaHasta": (future + timedelta(hours=2)).isoformat(),
            }
        )

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", fake_request)
    client = _build_client(monkeypatch)

    result = client.get_notification()

    assert result is not None
    assert result["mensaje"] == "Mantenimiento programado"
    assert result["activo"] is True
    assert result["vigencia"].isoformat() == future.astimezone(fixed_now.tzinfo).isoformat()
    assert result["vigencia_hasta"].isoformat() == (future + timedelta(hours=2)).astimezone(fixed_now.tzinfo).isoformat()


def test_get_notification_returns_none_when_inactive(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(self: iol_client_module.IOLClient, method: str, url: str, **_: Any) -> SimpleNamespace:
        return _response({"mensaje": "Aviso", "activo": False})

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", fake_request)
    client = _build_client(monkeypatch)

    assert client.get_notification() is None


def test_get_notification_returns_none_when_expired(monkeypatch: pytest.MonkeyPatch, fixed_now: datetime) -> None:
    expired = fixed_now - timedelta(hours=1)

    def fake_request(self: iol_client_module.IOLClient, method: str, url: str, **_: Any) -> SimpleNamespace:
        return _response({"mensaje": "Aviso", "activo": True, "vigenciaHasta": expired.isoformat()})

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", fake_request)
    client = _build_client(monkeypatch)

    assert client.get_notification() is None


def test_get_notification_parses_mixed_datetime_formats(monkeypatch: pytest.MonkeyPatch, fixed_now: datetime) -> None:
    epoch = int((fixed_now + timedelta(hours=3)).timestamp())

    def fake_request(self: iol_client_module.IOLClient, method: str, url: str, **_: Any) -> SimpleNamespace:
        return _response({"mensaje": "Aviso", "activo": True, "vigencia": epoch, "vigenciaHasta": None})

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", fake_request)
    client = _build_client(monkeypatch)

    result = client.get_notification()

    assert result is not None
    assert isinstance(result["vigencia"], datetime)
    assert result["vigencia"].timestamp() == pytest.approx(epoch)
    assert result["vigencia_hasta"] is None


def test_get_notification_uses_cache(monkeypatch: pytest.MonkeyPatch, fixed_now: datetime) -> None:
    calls: list[tuple[str, str]] = []

    def fake_request(self: iol_client_module.IOLClient, method: str, url: str, **_: Any) -> SimpleNamespace:
        calls.append((method, url))
        return _response({"mensaje": "Aviso", "activo": True, "vigencia": fixed_now.isoformat()})

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", fake_request)
    client = _build_client(monkeypatch)

    first = client.get_notification()
    second = client.get_notification()

    assert first == second
    assert len(calls) == 1


def test_get_notification_logs_without_message_body(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, fixed_now: datetime) -> None:
    caplog.set_level("INFO", logger=iol_client_module.logger.name)

    def fake_request(self: iol_client_module.IOLClient, method: str, url: str, **_: Any) -> SimpleNamespace:
        return _response({"mensaje": "Aviso confidencial", "activo": True, "vigencia": fixed_now.isoformat()})

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", fake_request)
    client = _build_client(monkeypatch)

    result = client.get_notification()

    assert result is not None
    records = [record for record in caplog.records if record.levelname == "INFO"]
    assert any("IOL notification fetch" in record.message for record in records)
    assert all("Aviso confidencial" not in record.message for record in records)


@pytest.mark.parametrize(
    "payload,status_code",
    [
        (None, 204),
        ({}, 200),
    ],
)
def test_get_notification_returns_none_on_empty_payload(
    monkeypatch: pytest.MonkeyPatch, payload: Any, status_code: int
) -> None:
    def fake_request(self: iol_client_module.IOLClient, method: str, url: str, **_: Any) -> SimpleNamespace:
        return _response(payload, status_code=status_code)

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", fake_request)
    client = _build_client(monkeypatch)

    assert client.get_notification() is None


@pytest.mark.parametrize("status_code", [403, 500])
def test_get_notification_handles_http_errors(monkeypatch: pytest.MonkeyPatch, status_code: int) -> None:
    error_response = SimpleNamespace(status_code=status_code)

    def fake_request(self: iol_client_module.IOLClient, method: str, url: str, **_: Any) -> None:
        raise requests.HTTPError(response=error_response)

    monkeypatch.setattr(iol_client_module.IOLClient, "_request", fake_request)
    client = _build_client(monkeypatch)

    assert client.get_notification() is None
