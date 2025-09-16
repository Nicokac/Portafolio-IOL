"""Tests for the legacy IOL client integration with bearer tokens."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

# Ensure the project root is importable regardless of pytest's invocation path.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infrastructure.iol.legacy import iol_client as iol_client_module


class FakeAuth:
    """Minimal auth stub exposing preloaded tokens."""

    def __init__(self) -> None:
        self.tokens = {
            "access_token": "access",
            "refresh_token": "refresh",
        }

    def auth_header(self) -> dict:
        raise AssertionError("auth_header should not be called in this test")

    def refresh(self) -> None:
        raise AssertionError("refresh should not be called in this test")


class StreamlitStub:
    """Streamlit stub exposing a mutable session_state mapping."""

    session_state: dict = {}


class StubIol:
    """Capture bearer assignment performed by the client."""

    def __init__(self, user: str, password: str) -> None:
        self.user = user
        self.password = password
        self.bearer: str | None = None
        self.refresh_token: str | None = None
        self.bearer_time: datetime | None = None
        self.gestionar_calls = 0

    def gestionar(self) -> None:
        self.gestionar_calls += 1


@pytest.fixture
def aware_moment() -> datetime:
    """Known aware timestamp used across the test."""

    return datetime(2024, 4, 1, 10, 45, tzinfo=ZoneInfo("UTC"))


def test_client_assigns_naive_bearer_time(monkeypatch: pytest.MonkeyPatch, aware_moment: datetime) -> None:
    """Instantiating IOLClient with bearer tokens should set a naive bearer_time."""

    monkeypatch.setattr(
        iol_client_module.TimeProvider,
        "now_datetime",
        classmethod(lambda cls: aware_moment),
    )
    monkeypatch.setattr(iol_client_module, "Iol", StubIol)
    monkeypatch.setattr(iol_client_module, "st", StreamlitStub)

    client = iol_client_module.IOLClient("user", "", auth=FakeAuth())

    assert isinstance(client.iol_market, StubIol)
    assert client.iol_market.bearer == "access"
    assert client.iol_market.refresh_token == "refresh"
    assert client.iol_market.bearer_time is not None
    assert client.iol_market.bearer_time.tzinfo is None
