import pytest
from types import SimpleNamespace
from pathlib import Path

from services import cache as svc_cache
from infrastructure.iol.auth import InvalidCredentialsError


def test_get_client_cached_clears_tokens_and_raises(monkeypatch, tmp_path):
    state = {}
    monkeypatch.setattr(svc_cache, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr("shared.cache.st", SimpleNamespace(session_state=state))
    built = {"called": False}

    tokens_path = tmp_path / "tokens.json"
    tokens_path.write_text("{}")

    class DummyAuth:
        def __init__(self, user, password, tokens_file=None, allow_plain_tokens=False):
            self.tokens_file = Path(tokens_file)

        def refresh(self):
            self.clear_tokens()
            raise InvalidCredentialsError()

        def clear_tokens(self):
            try:
                self.tokens_file.unlink()
            except FileNotFoundError:
                pass

    def dummy_build(user, password, tokens_file=None, auth=None):
        built["called"] = True
        return object()

    monkeypatch.setattr(svc_cache, "IOLAuth", DummyAuth)
    monkeypatch.setattr(svc_cache, "_build_iol_client", dummy_build)
    svc_cache.get_client_cached.clear()

    with pytest.raises(InvalidCredentialsError):
        svc_cache.get_client_cached("k", "u", tokens_path)

    assert not tokens_path.exists()
    assert built["called"] is False


def test_build_iol_client_triggers_logout(monkeypatch):
    state = {"IOL_USERNAME": "u"}
    monkeypatch.setattr(svc_cache, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr("shared.cache.st", SimpleNamespace(session_state=state))

    def dummy_get_client_cached(cache_key, user, tokens_file):
        raise InvalidCredentialsError()

    monkeypatch.setattr(svc_cache, "get_client_cached", dummy_get_client_cached)

    logout_called = {"user": None}

    def dummy_logout(user=None, password=""):
        logout_called["user"] = user
        state.clear()
        state["force_login"] = True

    monkeypatch.setattr("application.auth_service.logout", dummy_logout)

    cli, err = svc_cache.build_iol_client()

    assert cli is None
    assert isinstance(err, InvalidCredentialsError)
    assert logout_called["user"] is None
    assert state.get("force_login") is True


def test_get_client_cached_refresh_login_attempt(monkeypatch):
    state = {}
    monkeypatch.setattr(svc_cache, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr("shared.cache.st", SimpleNamespace(session_state=state))
    called = {"login": False}

    class DummyAuth:
        def __init__(self, user, password, tokens_file=None, allow_plain_tokens=False):
            pass

        def refresh(self):
            raise InvalidCredentialsError()

        def login(self):
            called["login"] = True
            raise InvalidCredentialsError()

        def clear_tokens(self):
            pass

    monkeypatch.setattr(svc_cache, "IOLAuth", DummyAuth)
    svc_cache.get_client_cached.clear()

    with pytest.raises(InvalidCredentialsError):
        svc_cache.get_client_cached("k", "u", None)

    assert called["login"] is False
