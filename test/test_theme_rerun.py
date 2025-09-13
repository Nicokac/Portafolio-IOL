import streamlit as st

from services import cache as svc_cache


def test_theme_change_reuses_client(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    st.session_state.update(
        {
            "session_id": "A",
            "IOL_USERNAME": "user",
            "authenticated": True,
            "client_salt": "s",
            "ui_theme": "light",
        }
    )

    created = []

    class DummyAuth:
        def __init__(self, *a, **k):
            self.tokens = {"access_token": "x", "refresh_token": "r"}

        def refresh(self):
            return self.tokens

    monkeypatch.setattr(svc_cache, "IOLAuth", DummyAuth)

    def dummy_build(user, password, tokens_file=None, auth=None):
        obj = object()
        created.append(obj)
        return obj

    monkeypatch.setattr(svc_cache, "_build_iol_client", dummy_build)

    cli1, err1 = svc_cache.build_iol_client()
    assert err1 is None
    assert len(created) == 1

    st.session_state["ui_theme"] = "dark"
    cli2, err2 = svc_cache.build_iol_client()
    assert err2 is None
    assert cli1 is cli2
    assert len(created) == 1
