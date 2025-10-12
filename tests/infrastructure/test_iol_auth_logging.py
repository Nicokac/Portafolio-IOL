import importlib
import logging

import pytest

from shared import config


@pytest.fixture(autouse=True)
def reset_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config.settings, "tokens_key", None, raising=False)
    monkeypatch.setattr(config.settings, "fastapi_tokens_key", None, raising=False)
    monkeypatch.setattr(config.settings, "allow_plain_tokens", False, raising=False)
    monkeypatch.setattr(config.settings, "app_env", "dev", raising=False)
    monkeypatch.delenv("IOL_TOKENS_KEY", raising=False)
    monkeypatch.delenv("FASTAPI_TOKENS_KEY", raising=False)
    yield


def _reload_auth_module():
    import infrastructure.iol.auth as auth_module

    return importlib.reload(auth_module)


def test_plain_tokens_emit_warning(tmp_path, caplog, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config.settings, "tokens_key", None, raising=False)
    monkeypatch.setattr(config.settings, "allow_plain_tokens", True, raising=False)
    auth_module = _reload_auth_module()
    target = tmp_path / "tokens.json"
    with caplog.at_level(logging.WARNING, logger="infrastructure.iol.auth"):
        auth = auth_module.IOLAuth("user", "pass", tokens_file=target, allow_plain_tokens=True)
        assert auth.allow_plain_tokens is True
    assert "Plain token storage enabled (development only)" in caplog.text


def test_plain_tokens_rejected_in_prod(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config.settings, "tokens_key", None, raising=False)
    monkeypatch.setattr(config.settings, "allow_plain_tokens", True, raising=False)
    monkeypatch.setattr(config.settings, "app_env", "prod", raising=False)
    auth_module = _reload_auth_module()
    target = tmp_path / "tokens.json"
    with pytest.raises(RuntimeError, match="Plain token storage cannot be enabled"):
        auth_module.IOLAuth("user", "pass", tokens_file=target, allow_plain_tokens=True)
