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
    first = svc_cache.get_client_cached("k", "u", "", None)
    again = svc_cache.get_client_cached("k", "u", "", None)
    assert first is again

    st.session_state["session_id"] = "B"
    other = svc_cache.get_client_cached("k", "u", "", None)
    assert other is not first

    st.session_state["session_id"] = "A"
    svc_cache.get_client_cached.clear("k")
    rebuilt = svc_cache.get_client_cached("k", "u", "", None)
    assert rebuilt is not first

    assert len(created) == 3


def test_build_iol_client_is_session_isolated(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    st.session_state.update({"IOL_USERNAME": "u"})

    created = []

    def dummy_build(user, password, tokens_file=None):
        obj = object()
        created.append(obj)
        return obj

    monkeypatch.setattr(svc_cache, "_build_iol_client", dummy_build)

    st.session_state["session_id"] = "A"
    svc_cache.get_client_cached.clear()
    cli_a, err_a = svc_cache.build_iol_client()
    assert err_a is None

    st.session_state["session_id"] = "B"
    svc_cache.get_client_cached.clear()
    cli_b, err_b = svc_cache.build_iol_client()
    assert err_b is None

    assert cli_a is not cli_b

    st.session_state["session_id"] = "A"
    cli_a2, _ = svc_cache.build_iol_client()
    assert cli_a2 is cli_a

    assert len(created) == 2
