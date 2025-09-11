import pytest
from unittest.mock import patch

from application import auth_service
from application.auth_service import AuthenticationError


def test_login_success():
    with patch("application.auth_service.IOLAuth") as mock_auth:
        mock_auth.return_value.login.return_value = {"access_token": "tok"}
        tokens = auth_service.login("user", "pass")
        assert tokens["access_token"] == "tok"
        mock_auth.assert_called_once_with("user", "pass")
        mock_auth.return_value.login.assert_called_once()


def test_login_invalid_raises():
    with patch("application.auth_service.IOLAuth") as mock_auth:
        mock_auth.return_value.login.return_value = {}
        with pytest.raises(AuthenticationError):
            auth_service.login("u", "p")


def test_logout_calls_clear_tokens():
    with patch("application.auth_service.IOLAuth") as mock_auth:
        auth_service.logout("user")
        mock_auth.assert_called_once_with("user", "")
        mock_auth.return_value.clear_tokens.assert_called_once()
