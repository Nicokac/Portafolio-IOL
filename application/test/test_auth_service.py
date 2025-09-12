import pytest
from unittest.mock import patch
from pathlib import Path

import streamlit as st

from application import auth_service
from application.auth_service import AuthenticationError


def test_login_success():
    with patch("application.auth_service.IOLAuth") as mock_auth:
        mock_auth.return_value.login.return_value = {"access_token": "tok"}
        tokens = auth_service.login("user", "pass")
        assert tokens["access_token"] == "tok"
        mock_auth.assert_called_once_with(
            "user",
            "pass",
            tokens_file=Path("tokens") / "user.json",
            allow_plain_tokens=False,
        )
        mock_auth.return_value.login.assert_called_once()


def test_login_invalid_raises():
    with patch("application.auth_service.IOLAuth") as mock_auth:
        mock_auth.return_value.login.return_value = {}
        with pytest.raises(AuthenticationError):
            auth_service.login("u", "p")
        mock_auth.assert_called_once_with(
            "u",
            "p",
            tokens_file=Path("tokens") / "u.json",
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
        auth_service.logout("user")
        mock_auth.assert_called_once_with(
            "user", "", tokens_file=Path("tokens") / "user.json"
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
