import hashlib
from pathlib import Path
import streamlit as st


class DummyAuth:
    """Simple auth mock storing tokens per file and tracking calls."""

    FILES: dict[Path, dict] = {}
    TOKEN_SEQ = 0

    def __init__(self, user, password, tokens_file=None, allow_plain_tokens=False):
        self.user = user
        self.password = password
        self.tokens_file = Path(tokens_file) if tokens_file else Path("tokens.json")
        self.tokens = DummyAuth.FILES.get(self.tokens_file, {}).copy()
        self.calls: list[str] = []

    def login(self) -> dict:
        DummyAuth.TOKEN_SEQ += 1
        self.calls.append("login")
        self.tokens = {
            "access_token": f"a{DummyAuth.TOKEN_SEQ}",
            "refresh_token": "r",
        }
        DummyAuth.FILES[self.tokens_file] = self.tokens.copy()
        return self.tokens

    def refresh(self) -> dict:
        DummyAuth.TOKEN_SEQ += 1
        self.calls.append("refresh")
        self.tokens["access_token"] = f"a{DummyAuth.TOKEN_SEQ}"
        DummyAuth.FILES[self.tokens_file] = self.tokens.copy()
        return self.tokens

    def clear_tokens(self) -> None:
        self.calls.append("clear")
        DummyAuth.FILES.pop(self.tokens_file, None)
        self.tokens = {}

    def auth_header(self) -> dict:
        token = self.tokens.get("access_token", "")
        return {"Authorization": f"Bearer {token}"} if token else {}


class DummyClient:
    def __init__(self, user, password, tokens_file=None):
        self.auth = DummyAuth(user, password, tokens_file)
        self.calls: list[str] = []

    def get_portfolio(self):
        self.calls.append("get_portfolio")
        if self.auth.tokens.get("access_token") == "expired":
            self.auth.refresh()
        return {"ok": 1}

def _reset_dummy():
    DummyAuth.FILES.clear()
    DummyAuth.TOKEN_SEQ = 0


def test_initial_login_generates_tokens(monkeypatch):
    from application import auth_service

    _reset_dummy()
    monkeypatch.setattr(st, "session_state", {})
    monkeypatch.setattr(auth_service, "IOLAuth", DummyAuth)

    tokens = auth_service.login("user", "pass")
    assert tokens["access_token"] == "a1"
    user_hash = hashlib.sha256("user".encode()).hexdigest()[:12]
    path = Path("tokens") / f"user-{user_hash}.json"
    assert DummyAuth.FILES[path] == tokens


def test_rerun_without_password_reuses_refresh_token(monkeypatch):
    from services import cache as svc_cache

    _reset_dummy()
    monkeypatch.setattr(st, "session_state", {})
    svc_cache.get_client_cached.clear()
    monkeypatch.setattr(svc_cache, "IOLAuth", DummyAuth)
    monkeypatch.setattr("application.auth_service.logout", lambda *a, **k: None)

    path = Path("tokens") / "u.json"
    st.session_state.update(
        {"IOL_USERNAME": "user", "client_salt": "s", "tokens_file": str(path)}
    )
    DummyAuth.FILES[path] = {"access_token": "old", "refresh_token": "r"}

    def dummy_build(user, password, tokens_file=None, auth=None):
        cli = DummyClient(user, password, tokens_file)
        if auth is not None:
            cli.auth = auth
        return cli

    monkeypatch.setattr(svc_cache, "_build_iol_client", dummy_build)
    monkeypatch.setattr(svc_cache, "IOLAuth", DummyAuth)

    cli, err = svc_cache.build_iol_client()
    assert err is None
    assert cli.auth.tokens["refresh_token"] == "r"
    assert cli.auth.calls == ["refresh"]


def test_expired_bearer_triggers_refresh(monkeypatch):
    from services import cache as svc_cache

    _reset_dummy()
    monkeypatch.setattr(st, "session_state", {})
    svc_cache.get_client_cached.clear()
    monkeypatch.setattr(svc_cache, "IOLAuth", DummyAuth)
    monkeypatch.setattr("application.auth_service.logout", lambda *a, **k: None)

    path = Path("tokens") / "u.json"
    st.session_state.update(
        {"IOL_USERNAME": "user", "client_salt": "s", "tokens_file": str(path)}
    )
    DummyAuth.FILES[path] = {"access_token": "expired", "refresh_token": "r"}

    def dummy_build(user, password, tokens_file=None, auth=None):
        cli = DummyClient(user, password, tokens_file)
        if auth is not None:
            cli.auth = auth
        return cli

    monkeypatch.setattr(svc_cache, "_build_iol_client", dummy_build)
    monkeypatch.setattr(svc_cache, "IOLAuth", DummyAuth)

    cli, err = svc_cache.build_iol_client()
    assert err is None
    assert cli.auth.calls == ["refresh"]
    cli.get_portfolio()
    assert cli.auth.calls == ["refresh"]
    assert cli.auth.tokens["access_token"] == "a1"
    assert cli.auth.tokens["refresh_token"] == "r"


def test_logout_clears_tokens_and_allows_clean_login(monkeypatch):
    from application import auth_service

    _reset_dummy()
    monkeypatch.setattr(st, "session_state", {})
    monkeypatch.setattr(auth_service, "IOLAuth", DummyAuth)

    tokens1 = auth_service.login("user", "pass")
    user_hash = hashlib.sha256("user".encode()).hexdigest()[:12]
    path = Path("tokens") / f"user-{user_hash}.json"
    assert DummyAuth.FILES[path] == tokens1

    monkeypatch.setattr(st, "rerun", lambda *a, **k: None)
    auth_service.logout("user")
    assert path not in DummyAuth.FILES
    assert st.session_state.get("force_login") is True
    assert st.session_state.get("logout_done") is True

    tokens2 = auth_service.login("user", "pass")
    assert tokens2["access_token"] == "a2"
    assert DummyAuth.FILES[path] == tokens2
