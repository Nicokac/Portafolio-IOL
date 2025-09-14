import types
import pytest
from shared import config


def test_secret_or_env_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("SOME_KEY", "env_value")
    monkeypatch.setattr(config, "st", types.SimpleNamespace())
    assert config.settings.secret_or_env("SOME_KEY") == "env_value"


def test_secret_or_env_returns_default(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    monkeypatch.setattr(config, "st", types.SimpleNamespace())
    assert config.settings.secret_or_env("MISSING_KEY", default="default") == "default"


def test_ensure_tokens_key_requires_key(monkeypatch):
    dummy = types.SimpleNamespace(tokens_key="", allow_plain_tokens=False)
    monkeypatch.setattr(config, "settings", dummy)
    with pytest.raises(SystemExit):
        config.ensure_tokens_key()


def test_ensure_tokens_key_allowed(monkeypatch):
    dummy = types.SimpleNamespace(tokens_key="", allow_plain_tokens=True)
    monkeypatch.setattr(config, "settings", dummy)
    config.ensure_tokens_key()