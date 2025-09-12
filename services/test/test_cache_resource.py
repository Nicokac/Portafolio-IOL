import streamlit as st
from services import cache as svc_cache


def test_get_client_cached_is_session_isolated(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})

    created = []

    def dummy_build(user, password, tokens_file=None):
        obj = object()
        created.append(obj)
        return obj

    monkeypatch.setattr(svc_cache, "_build_iol_client", dummy_build)

    st.session_state["session_id"] = "A"
    first = svc_cache.get_client_cached("k", "u", "p", None)
    again = svc_cache.get_client_cached("k", "u", "p", None)
    assert first is again

    st.session_state["session_id"] = "B"
    other = svc_cache.get_client_cached("k", "u", "p", None)
    assert other is not first

    st.session_state["session_id"] = "A"
    svc_cache.get_client_cached.clear("k")
    rebuilt = svc_cache.get_client_cached("k", "u", "p", None)
    assert rebuilt is not first

    assert len(created) == 3
