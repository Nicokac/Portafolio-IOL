import pytest
from unittest.mock import patch
from pathlib import Path
import hashlib

import streamlit as st

from application import auth_service
from application.auth_service import AuthenticationError


def test_login_success(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    with patch("application.auth_service.IOLAuth") as mock_auth:
        mock_auth.return_value.login.return_value = {"access_token": "tok"}
        user = "user"
        tokens = auth_service.login(user, "pass")
        assert tokens["access_token"] == "tok"
        assert "client_salt" in st.session_state
        assert len(st.session_state["client_salt"]) == 32
        user_hash = hashlib.sha256(user.encode()).hexdigest()[:8]
        expected = Path("tokens") / f"{user}-{user_hash}.json"
        mock_auth.assert_called_once_with(
            user,
            "pass",
            tokens_file=expected,
            allow_plain_tokens=False,
        )
        mock_auth.return_value.login.assert_called_once()


def test_login_invalid_raises(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    with patch("application.auth_service.IOLAuth") as mock_auth:
        mock_auth.return_value.login.return_value = {}
        user = "u"
        with pytest.raises(AuthenticationError):
            auth_service.login(user, "p")
        assert "client_salt" not in st.session_state
        user_hash = hashlib.sha256(user.encode()).hexdigest()[:8]
        expected = Path("tokens") / f"{user}-{user_hash}.json"
        mock_auth.assert_called_once_with(
            user,
            "p",
            tokens_file=expected,
            allow_plain_tokens=False,
        )


def test_logout_clears_session_state(monkeypatch):
    monkeypatch.setattr(
        st,
        "session_state",
        {
            "session_id": "A",
            "IOL_USERNAME": "user",
            "authenticated": True,
            "client_salt": "s",
            "tokens_file": "foo",
            "x": 1,
        },
    )
    monkeypatch.setattr(st, "rerun", lambda *a, **k: None)
    with patch("application.auth_service.IOLAuth") as mock_auth:
        user = "user"
        auth_service.logout(user)
        user_hash = hashlib.sha256(user.encode()).hexdigest()[:8]
        expected = Path("tokens") / f"{user}-{user_hash}.json"
        mock_auth.assert_called_once_with(
            "user",
            "",
            tokens_file=expected,
            allow_plain_tokens=False,
        )
        mock_auth.return_value.clear_tokens.assert_called_once()
    assert st.session_state.get("force_login") is True
    assert st.session_state.get("logout_done") is True


def test_logout_generates_new_session_id(monkeypatch):
    from types import SimpleNamespace
    import app

    monkeypatch.setattr(st, "session_state", {"session_id": "old"})
    monkeypatch.setattr(st, "rerun", lambda *a, **k: None)
    with patch("application.auth_service.IOLAuth"):
        auth_service.logout("user")
    assert st.session_state.get("force_login") is True
    assert st.session_state.get("logout_done") is True

    monkeypatch.setattr(app, "render_login_page", lambda: None)
    monkeypatch.setattr(app, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(app, "render_action_menu", lambda: None)
    monkeypatch.setattr(app, "render_footer", lambda: None)
    monkeypatch.setattr(app, "render_portfolio_section", lambda *a, **k: None)
    monkeypatch.setattr(app, "get_fx_rates_cached", lambda: (None, None))
    monkeypatch.setattr(app, "build_iol_client", lambda: object())
    monkeypatch.setattr(app, "configure_logging", lambda level=None, json_format=None: None)
    monkeypatch.setattr(app, "ensure_tokens_key", lambda: None)
    monkeypatch.setattr(app, "_parse_args", lambda argv: SimpleNamespace(log_level=None, log_format=None))
    monkeypatch.setattr(app, "uuid4", lambda: SimpleNamespace(hex="new_session"))
    monkeypatch.setattr(st, "stop", lambda: (_ for _ in ()).throw(SystemExit))

    with pytest.raises(SystemExit):
        app.main([])

    assert st.session_state.get("session_id") == "new_session"
