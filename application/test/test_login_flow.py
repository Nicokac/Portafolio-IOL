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
    st.session_state.clear()
    st.session_state["IOL_USERNAME"] = "user"
    st.session_state["IOL_PASSWORD"] = "pass"
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

    assert st.session_state.get("force_login") is True

    with patch("ui.login.render_login_page") as mock_login:
        from ui.login import render_login_page
        if st.session_state.get("force_login"):
            render_login_page()
        mock_login.assert_called_once()

    st.session_state.clear()