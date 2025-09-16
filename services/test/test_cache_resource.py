import pytest
import streamlit as st
from shared.cache import cache
from shared.errors import InvalidCredentialsError
from services import cache as svc_cache


def test_get_client_cached_is_session_isolated(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})

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

    st.session_state["session_id"] = "A"
    first = svc_cache.get_client_cached("k", "u", None)
    again = svc_cache.get_client_cached("k", "u", None)
    assert first is again

    st.session_state["session_id"] = "B"
    other = svc_cache.get_client_cached("k", "u", None)
    assert other is not first

    st.session_state["session_id"] = "A"
    svc_cache.get_client_cached.clear("k", "u", None)
    rebuilt = svc_cache.get_client_cached("k", "u", None)
    assert rebuilt is not first

    assert len(created) == 3


def test_build_iol_client_is_session_isolated(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    st.session_state.update({"IOL_USERNAME": "u"})

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


def test_get_client_cached_invalid_credentials(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    svc_cache.get_client_cached.clear()

    cleared = {"called": False}

    class DummyAuth:
        def __init__(self, *a, **k):
            pass

        def refresh(self):
            raise InvalidCredentialsError()

        def clear_tokens(self):
            cleared["called"] = True

    monkeypatch.setattr(svc_cache, "IOLAuth", DummyAuth)

    with pytest.raises(InvalidCredentialsError):
        svc_cache.get_client_cached("k", "u", None)

    assert st.session_state["force_login"] is True
    assert cleared["called"] is True


def test_cache_resource_uses_all_arguments(monkeypatch):
    monkeypatch.setattr(st, "session_state", {"session_id": "SID"})

    created = []

    @cache.cache_resource
    def builder(prefix: str, value: int, *, toggle: bool = False):
        obj = object()
        created.append((prefix, value, toggle, obj))
        return obj

    first = builder("same", 1, toggle=False)
    again = builder("same", 1, toggle=False)
    assert first is again

    other_kw = builder("same", 1, toggle=True)
    assert other_kw is not first

    other_pos = builder("same", 2, toggle=False)
    assert other_pos is not first
    assert other_pos is not other_kw

    assert len(created) == 3

    builder.clear("same", 1, toggle=False)
    rebuilt = builder("same", 1, toggle=False)
    assert rebuilt is not first

    # Re-using kwargs order shouldn't create duplicate entries.
    again_kw = builder("same", 1, toggle=True)
    assert again_kw is other_kw
