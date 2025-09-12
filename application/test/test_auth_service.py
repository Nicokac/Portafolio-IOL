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


def test_logout_clears_only_auth_keys(monkeypatch):
    monkeypatch.setattr(
        st,
        "session_state",
        {
            "session_id": "A",
            "IOL_USERNAME": "user",
            "IOL_PASSWORD": "pass",
            "authenticated": True,
            "client_salt": "s",
            "tokens_file": "foo",
            "x": 1,
        },
    )
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
    assert st.session_state == {"session_id": "A", "x": 1}

def test_logout_is_session_isolated(monkeypatch):
    other = {"data": 42}
    monkeypatch.setattr(
        st,
        "session_state",
        {
            "session_id": "A",
            "IOL_USERNAME": "user",
            "authenticated": True,
            "other": other,
        },
    )
    with patch("application.auth_service.IOLAuth"):
        auth_service.logout("user")
    assert st.session_state.get("other") is other
    assert "IOL_USERNAME" not in st.session_state
