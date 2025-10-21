"""Tests covering the shared FakeAuth fixture."""

from tests.fixtures.auth import FakeAuth


def test_fake_auth_basic(fake_auth):
    auth = fake_auth
    assert auth.access_token == "FAKE_TOKEN"
    assert auth.refresh_token == "REFRESH_TOKEN"
    assert auth.tokens["access_token"] == "FAKE_TOKEN"
    assert auth.tokens["refresh_token"] == "REFRESH_TOKEN"
    assert not auth.is_expired()

    auth.mark_expired()

    assert auth.is_expired()


def test_fake_auth_custom_tokens():
    auth = FakeAuth(access="ACCESS", refresh="REFRESH", expired=True)
    assert auth.access_token == "ACCESS"
    assert auth.refresh_token == "REFRESH"
    assert auth.tokens == {"access_token": "ACCESS", "refresh_token": "REFRESH"}
    assert auth.is_expired()
