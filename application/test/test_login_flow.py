import streamlit as st
from ui.actions import render_action_menu
from unittest.mock import patch


class DummyCtx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def button(self, *args, **kwargs):
        return False


def test_logout_forces_login_page(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    st.session_state["IOL_USERNAME"] = "user"
    st.session_state["authenticated"] = True
    st.session_state["logout_pending"] = True

    monkeypatch.setattr(st, "popover", lambda *a, **k: DummyCtx())
    monkeypatch.setattr(st, "columns", lambda n: [DummyCtx(), DummyCtx()])
    monkeypatch.setattr(st, "spinner", lambda *a, **k: DummyCtx())
    monkeypatch.setattr(st, "toast", lambda *a, **k: None)
    monkeypatch.setattr(st, "success", lambda *a, **k: None)
    monkeypatch.setattr(st, "error", lambda *a, **k: None)
    monkeypatch.setattr(st, "stop", lambda *a, **k: None)
    monkeypatch.setattr(st, "rerun", lambda *a, **k: None)
    monkeypatch.setattr(st, "caption", lambda *a, **k: None)

    with patch("ui.actions.IOLAuth") as mock_auth:
        mock_auth.return_value.clear_tokens.return_value = None
        render_action_menu()
        mock_auth.assert_called_once_with("user", "", tokens_file=None)

    assert st.session_state.get("force_login") is True

    with patch("ui.login.render_login_page") as mock_login:
        from ui.login import render_login_page
        if st.session_state.get("force_login"):
            render_login_page()
        mock_login.assert_called_once()

    st.session_state.clear()

def test_service_build_iol_client_returns_error(monkeypatch):
    from services import cache as svc_cache

    monkeypatch.setattr(st, "session_state", {})
    st.session_state["IOL_USERNAME"] = "user"
    st.session_state["IOL_PASSWORD"] = "pass"

    def boom(*args, **kwargs):
        raise RuntimeError("bad creds")

    monkeypatch.setattr(svc_cache, "get_client_cached", boom)

    cli, error = svc_cache.build_iol_client()

    assert cli is None
    assert str(error) == "bad creds"


def test_controller_build_iol_client_handles_error(monkeypatch):
    from controllers import auth

    monkeypatch.setattr(st, "session_state", {})
    st.session_state["IOL_PASSWORD"] = "pass"

    monkeypatch.setattr(auth, "_build_iol_client", lambda: (None, RuntimeError("bad creds")))
    monkeypatch.setattr(st, "rerun", lambda *a, **k: None)

    auth.build_iol_client()

    assert st.session_state.get("force_login") is True
    assert st.session_state.get("login_error") == "Error de conexión"
    assert st.session_state.get("IOL_PASSWORD") == ""


def test_controller_build_iol_client_success_clears_password(monkeypatch):
    from controllers import auth

    monkeypatch.setattr(st, "session_state", {})
    st.session_state["IOL_PASSWORD"] = "pass"

    dummy_cli = object()
    monkeypatch.setattr(auth, "_build_iol_client", lambda: (dummy_cli, None))

    cli = auth.build_iol_client()

    assert cli is dummy_cli
    assert st.session_state.get("authenticated") is True
    assert "IOL_PASSWORD" not in st.session_state


def test_render_login_page_shows_error(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    st.session_state["login_error"] = "fail"

    monkeypatch.setattr("ui.login.render_header", lambda: None)
    monkeypatch.setattr(st, "form", lambda *a, **k: DummyCtx())
    monkeypatch.setattr(st, "text_input", lambda *a, **k: None)
    monkeypatch.setattr(st, "form_submit_button", lambda *a, **k: False)
    monkeypatch.setattr(st, "caption", lambda *a, **k: None)

    captured = {}
    monkeypatch.setattr(st, "error", lambda msg: captured.setdefault("msg", msg))

    from ui.login import render_login_page

    render_login_page()

    assert captured.get("msg") == "fail"
