"""Tests for the compatibility session authentication backoff handling."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from infrastructure.iol.compat import session as compat_session_module
from shared.errors import InvalidCredentialsError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def compat_session(monkeypatch: pytest.MonkeyPatch):
    """Provide a freshly initialised compat session with a stubbed Streamlit."""

    stub = SimpleNamespace(session_state={})
    monkeypatch.setattr(compat_session_module, "st", stub)
    compat_session_module.LegacySession._instance = None
    session = compat_session_module.LegacySession()
    try:
        yield session
    finally:
        compat_session_module.LegacySession._instance = None


def test_ensure_authenticated_skips_when_auth_unavailable(
    compat_session: compat_session_module.LegacySession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once marked unavailable, ensure_authenticated should not retry without changes."""

    builder = MagicMock(side_effect=InvalidCredentialsError("boom"))
    monkeypatch.setattr(compat_session, "_build_session", builder)

    first = compat_session.ensure_authenticated("user", "pass", auth=None)
    assert first is None
    assert compat_session.is_auth_unavailable() is True
    assert compat_session_module.st.session_state["legacy_auth_unavailable"] is True
    assert builder.call_count == 1

    second = compat_session.ensure_authenticated("user", "pass", auth=None)
    assert second is None
    assert compat_session.is_auth_unavailable() is True
    assert builder.call_count == 1


def test_fetch_with_backoff_retries_after_password_change(
    compat_session: compat_session_module.LegacySession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Password changes should reset the unavailable flag and allow another attempt."""

    builder = MagicMock(side_effect=InvalidCredentialsError("boom"))
    monkeypatch.setattr(compat_session, "_build_session", builder)

    data, auth_failed = compat_session.fetch_with_backoff(
        "bcba",
        "GGAL",
        auth_user="user",
        auth_password="pass",
        auth=None,
    )
    assert data is None and auth_failed is True
    assert compat_session_module.st.session_state["legacy_auth_unavailable"] is True
    assert builder.call_count == 1

    data, auth_failed = compat_session.fetch_with_backoff(
        "bcba",
        "GGAL",
        auth_user="user",
        auth_password="pass",
        auth=None,
    )
    assert data is None and auth_failed is True
    assert builder.call_count == 1

    data, auth_failed = compat_session.fetch_with_backoff(
        "bcba",
        "GGAL",
        auth_user="user",
        auth_password="new-pass",
        auth=None,
    )
    assert data is None and auth_failed is True
    assert builder.call_count == 2


def test_fetch_with_backoff_retries_after_token_change(
    compat_session: compat_session_module.LegacySession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Token updates should reset the unavailable flag and trigger a new attempt."""

    builder = MagicMock(side_effect=InvalidCredentialsError("boom"))
    monkeypatch.setattr(compat_session, "_build_session", builder)

    auth = SimpleNamespace(tokens={"access_token": "a", "refresh_token": "b"})
    data, auth_failed = compat_session.fetch_with_backoff(
        "bcba",
        "GGAL",
        auth_user="user",
        auth_password="",
        auth=auth,
    )
    assert data is None and auth_failed is True
    assert builder.call_count == 1

    data, auth_failed = compat_session.fetch_with_backoff(
        "bcba",
        "GGAL",
        auth_user="user",
        auth_password="",
        auth=auth,
    )
    assert data is None and auth_failed is True
    assert builder.call_count == 1

    new_auth = SimpleNamespace(tokens={"access_token": "new", "refresh_token": "c"})
    data, auth_failed = compat_session.fetch_with_backoff(
        "bcba",
        "GGAL",
        auth_user="user",
        auth_password="",
        auth=new_auth,
    )
    assert data is None and auth_failed is True
    assert builder.call_count == 2


def test_token_rotation_does_not_retry_until_values_change(
    compat_session: compat_session_module.LegacySession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Token refreshes with identical values must honour the sticky failure flag."""

    builder = MagicMock(side_effect=InvalidCredentialsError("boom"))
    monkeypatch.setattr(compat_session, "_build_session", builder)

    stale_tokens = {"access_token": "tok", "refresh_token": "ref"}
    auth = SimpleNamespace(tokens=stale_tokens)

    assert compat_session.ensure_authenticated("user", "", auth=auth) is None
    assert builder.call_count == 1

    rotated_auth = SimpleNamespace(tokens={"access_token": "tok", "refresh_token": "ref"})
    assert compat_session.ensure_authenticated("user", "", auth=rotated_auth) is None
    assert builder.call_count == 1

    fresh_auth = SimpleNamespace(tokens={"access_token": "tok-2", "refresh_token": "ref-2"})
    assert compat_session.ensure_authenticated("user", "", auth=fresh_auth) is None
    assert builder.call_count == 2


def test_sticky_flag_can_expire_after_cooldown(
    compat_session: compat_session_module.LegacySession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the configured cooldown the session may retry even without changes."""

    builder = MagicMock(side_effect=InvalidCredentialsError("boom"))
    monkeypatch.setattr(compat_session, "_build_session", builder)

    monkeypatch.setattr(compat_session_module, "AUTH_FAILURE_COOLDOWN_SECONDS", 10, raising=False)

    now = 100.0
    monkeypatch.setattr(compat_session_module.time, "monotonic", lambda: now)

    assert compat_session.ensure_authenticated("user", "pass", auth=None) is None
    assert builder.call_count == 1

    now += 5
    assert compat_session.ensure_authenticated("user", "pass", auth=None) is None
    assert builder.call_count == 1

    now += 6
    assert compat_session.ensure_authenticated("user", "pass", auth=None) is None
    assert builder.call_count == 2
