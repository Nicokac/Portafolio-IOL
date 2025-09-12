import requests
import streamlit as st
from types import SimpleNamespace


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
