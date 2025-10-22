"""Tests for ``LegacySession`` authentication backoff handling."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from infrastructure.iol.legacy import session as legacy_session_module
from shared.errors import InvalidCredentialsError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    assert legacy_session_module.st.session_state["legacy_auth_unavailable"] is True
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
    assert legacy_session_module.st.session_state["legacy_auth_unavailable"] is True
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


def test_token_rotation_does_not_retry_until_values_change(
    legacy_session: legacy_session_module.LegacySession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Token refreshes with identical values must honour the sticky failure flag."""

    builder = MagicMock(side_effect=InvalidCredentialsError("boom"))
    monkeypatch.setattr(legacy_session, "_build_session", builder)

    stale_tokens = {"access_token": "tok", "refresh_token": "ref"}
    auth = SimpleNamespace(tokens=stale_tokens)

    assert legacy_session.ensure_authenticated("user", "", auth=auth) is None
    assert builder.call_count == 1

    # Simulate a rotation that reuses the same token values (common during refresh
    # retries).  No additional authentication attempts should be triggered.
    rotated_auth = SimpleNamespace(tokens={"access_token": "tok", "refresh_token": "ref"})
    assert legacy_session.ensure_authenticated("user", "", auth=rotated_auth) is None
    assert builder.call_count == 1

    # Once the token payload actually changes, the sticky flag should be reset and
    # a new attempt may occur.
    fresh_auth = SimpleNamespace(tokens={"access_token": "tok-2", "refresh_token": "ref-2"})
    assert legacy_session.ensure_authenticated("user", "", auth=fresh_auth) is None
    assert builder.call_count == 2


def test_sticky_flag_can_expire_after_cooldown(
    legacy_session: legacy_session_module.LegacySession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the configured cooldown the session may retry even without changes."""

    builder = MagicMock(side_effect=InvalidCredentialsError("boom"))
    monkeypatch.setattr(legacy_session, "_build_session", builder)

    monkeypatch.setattr(legacy_session_module, "AUTH_FAILURE_COOLDOWN_SECONDS", 10, raising=False)

    # Start with a deterministic timestamp.
    now = 100.0
    monkeypatch.setattr(legacy_session_module.time, "monotonic", lambda: now)

    assert legacy_session.ensure_authenticated("user", "pass", auth=None) is None
    assert builder.call_count == 1

    # Before the cooldown expires the attempt should be short-circuited.
    now += 5
    assert legacy_session.ensure_authenticated("user", "pass", auth=None) is None
    assert builder.call_count == 1

    # After the cooldown expires another attempt should be permitted.
    now += 6
    assert legacy_session.ensure_authenticated("user", "pass", auth=None) is None
    assert builder.call_count == 2
