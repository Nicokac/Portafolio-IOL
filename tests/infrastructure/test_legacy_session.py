"""Tests for ``LegacySession`` authentication backoff handling."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infrastructure.iol.legacy import session as legacy_session_module
from shared.errors import InvalidCredentialsError


@pytest.fixture
def legacy_session(monkeypatch: pytest.MonkeyPatch):
    """Provide a freshly initialised ``LegacySession`` with a stubbed Streamlit."""

    stub = SimpleNamespace(session_state={})
    monkeypatch.setattr(legacy_session_module, "st", stub)
    legacy_session_module.LegacySession._instance = None
    session = legacy_session_module.LegacySession()
    try:
        yield session
    finally:
        legacy_session_module.LegacySession._instance = None


def test_ensure_authenticated_skips_when_auth_unavailable(
    legacy_session: legacy_session_module.LegacySession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once marked unavailable, ensure_authenticated should not retry without changes."""

    builder = MagicMock(side_effect=InvalidCredentialsError("boom"))
    monkeypatch.setattr(legacy_session, "_build_session", builder)

    first = legacy_session.ensure_authenticated("user", "pass", auth=None)
    assert first is None
    assert legacy_session.is_auth_unavailable() is True
    assert builder.call_count == 1

    second = legacy_session.ensure_authenticated("user", "pass", auth=None)
    assert second is None
    assert legacy_session.is_auth_unavailable() is True
    assert builder.call_count == 1


def test_fetch_with_backoff_retries_after_password_change(
    legacy_session: legacy_session_module.LegacySession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Password changes should reset the unavailable flag and allow another attempt."""

    builder = MagicMock(side_effect=InvalidCredentialsError("boom"))
    monkeypatch.setattr(legacy_session, "_build_session", builder)

    data, auth_failed = legacy_session.fetch_with_backoff(
        "bcba",
        "GGAL",
        auth_user="user",
        auth_password="pass",
        auth=None,
    )
    assert data is None and auth_failed is True
    assert builder.call_count == 1

    data, auth_failed = legacy_session.fetch_with_backoff(
        "bcba",
        "GGAL",
        auth_user="user",
        auth_password="pass",
        auth=None,
    )
    assert data is None and auth_failed is True
    assert builder.call_count == 1

    data, auth_failed = legacy_session.fetch_with_backoff(
        "bcba",
        "GGAL",
        auth_user="user",
        auth_password="new-pass",
        auth=None,
    )
    assert data is None and auth_failed is True
    assert builder.call_count == 2


def test_fetch_with_backoff_retries_after_token_change(
    legacy_session: legacy_session_module.LegacySession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Token updates should reset the unavailable flag and trigger a new attempt."""

    builder = MagicMock(side_effect=InvalidCredentialsError("boom"))
    monkeypatch.setattr(legacy_session, "_build_session", builder)

    auth = SimpleNamespace(tokens={"access_token": "a", "refresh_token": "b"})
    data, auth_failed = legacy_session.fetch_with_backoff(
        "bcba",
        "GGAL",
        auth_user="user",
        auth_password="",
        auth=auth,
    )
    assert data is None and auth_failed is True
    assert builder.call_count == 1

    data, auth_failed = legacy_session.fetch_with_backoff(
        "bcba",
        "GGAL",
        auth_user="user",
        auth_password="",
        auth=auth,
    )
    assert data is None and auth_failed is True
    assert builder.call_count == 1

    new_auth = SimpleNamespace(tokens={"access_token": "new", "refresh_token": "c"})
    data, auth_failed = legacy_session.fetch_with_backoff(
        "bcba",
        "GGAL",
        auth_user="user",
        auth_password="",
        auth=new_auth,
    )
    assert data is None and auth_failed is True
    assert builder.call_count == 2

