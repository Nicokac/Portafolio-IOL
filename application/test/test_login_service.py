from application.login_service import clear_password_keys, validate_tokens_key
from shared.config import settings


def test_validate_tokens_key_ok(monkeypatch):
    monkeypatch.setattr(settings, "tokens_key", "secret")
    result = validate_tokens_key()
    assert result.can_proceed is True
    assert result.level is None
    assert result.message is None


def test_validate_tokens_key_warning(monkeypatch):
    monkeypatch.setattr(settings, "tokens_key", "")
    monkeypatch.setattr(settings, "allow_plain_tokens", True)
    result = validate_tokens_key()
    assert result.can_proceed is True
    assert result.level == "warning"
    assert "sin cifrar" in result.message


def test_validate_tokens_key_error(monkeypatch):
    monkeypatch.setattr(settings, "tokens_key", "")
    monkeypatch.setattr(settings, "allow_plain_tokens", False)
    result = validate_tokens_key()
    assert result.can_proceed is False
    assert result.level == "error"
    assert "no está configurada" in result.message


def test_clear_password_keys_removes_sensitive_entries():
    state = {"foo": 1, "Password": "x", "api_password_token": "y", "bar": 2}
    clear_password_keys(state)
    assert "Password" not in state
    assert "api_password_token" not in state
    assert state == {"foo": 1, "bar": 2}
