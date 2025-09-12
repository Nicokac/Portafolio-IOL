import requests
import streamlit as st
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_refresh_flow_uses_refresh_token(monkeypatch):
    from infrastructure.iol.legacy.iol_client import IOLClient

    class DummyAuth:
        def __init__(self, user, password, tokens_file=None, allow_plain_tokens=False):
            self.tokens = {"access_token": "old", "refresh_token": "r"}
            self.calls = []

        def auth_header(self):
            return {"Authorization": f"Bearer {self.tokens['access_token']}"}

        def refresh(self):
            self.calls.append("refresh")
            self.tokens["access_token"] = "new"

        def login(self):
            self.calls.append("login")
            return self.tokens

    monkeypatch.setattr("infrastructure.iol.legacy.iol_client.IOLAuth", DummyAuth)
    monkeypatch.setattr(IOLClient, "_ensure_market_auth", lambda self: None)

    calls = {"n": 0}

    responses = [
        SimpleNamespace(status_code=401),
        SimpleNamespace(
            status_code=200,
            json=lambda: {"ok": 1},
            raise_for_status=lambda: None,
        ),
    ]

    def fake_request(self, method, url, headers=None, timeout=None, **kwargs):
        calls["n"] += 1
        resp = responses.pop(0)
        if not hasattr(resp, "raise_for_status"):
            def raise_for_status():
                raise requests.HTTPError(response=SimpleNamespace(status_code=resp.status_code))
            resp.raise_for_status = raise_for_status
        return resp

    monkeypatch.setattr(requests.Session, "request", fake_request, raising=False)

    cli = IOLClient("u", "p")
    r = cli._request("GET", "http://example.com")
    assert r.status_code == 200
    assert cli.auth.calls == ["refresh"]
    assert calls["n"] == 2


def test_rerun_preserves_session(monkeypatch):
    monkeypatch.setattr(st, "session_state", {"tokens": {"a": 1}, "session_id": "sid"})
    called = {}

    def fake_rerun():
        called["ok"] = True

    monkeypatch.setattr(st, "rerun", fake_rerun)
    st.rerun()
    assert called.get("ok") is True
    assert st.session_state["tokens"] == {"a": 1}
    assert st.session_state["session_id"] == "sid"

def test_invalid_refresh_token_forces_login(monkeypatch):
    from controllers import auth
    from infrastructure.iol.auth import InvalidCredentialsError
    mock_st = SimpleNamespace(session_state={"tokens": {"x": 1}}, rerun=MagicMock())
    monkeypatch.setattr(auth, "st", mock_st)

    class DummyProvider:
        def build_client(self):
            mock_st.session_state.pop("tokens", None)
            return None, InvalidCredentialsError("bad refresh")

    monkeypatch.setattr(auth, "get_auth_provider", lambda: DummyProvider())
    auth.build_iol_client()

    assert mock_st.session_state.get("force_login") is True
    assert "tokens" not in mock_st.session_state
    assert mock_st.rerun.called


def test_login_refresh_valid_then_invalid(monkeypatch):
    from services import cache as svc_cache

    class DummyAuth:
        def __init__(self, user, password, tokens_file=None, allow_plain_tokens=False):
            self.tokens = {"access_token": "a1", "refresh_token": "r"}
            self.calls = []

        def refresh(self):
            self.calls.append("refresh")
            self.tokens["access_token"] = "a2"

        def clear_tokens(self):
            self.tokens = {}

    class DummyClient:
        def __init__(self, user, password, tokens_file=None):
            self.auth = DummyAuth(user, password, tokens_file)

        def get_portfolio(self):
            if self.auth.tokens.get("access_token") == "expired":
                self.auth.refresh()
            return {"ok": 1}

    monkeypatch.setattr(svc_cache, "IOLAuth", DummyAuth)
    monkeypatch.setattr(svc_cache, "_build_iol_client", lambda u, p, tokens_file=None, auth=None: DummyClient(u, p, tokens_file))
    svc_cache.get_client_cached.clear()
    monkeypatch.setattr(svc_cache, "st", SimpleNamespace(session_state={}))
    svc_cache.st.session_state.update({"IOL_USERNAME": "u", "client_salt": "s", "tokens_file": "t.json"})

    cli, err = svc_cache.build_iol_client()
    assert err is None

    svc_cache.fetch_portfolio.clear()
    cli.auth.tokens["access_token"] = "expired"
    svc_cache.fetch_portfolio(cli)
    assert cli.auth.calls == ["refresh"]

    def bad_refresh():
        cli.auth.clear_tokens()
        raise svc_cache.InvalidCredentialsError("bad refresh")

    cli.auth.refresh = bad_refresh
    cli.auth.tokens["access_token"] = "expired"
    svc_cache.fetch_portfolio.clear()
    svc_cache.fetch_portfolio(cli)

    assert svc_cache.st.session_state.get("force_login") is True
    assert cli.auth.tokens == {}