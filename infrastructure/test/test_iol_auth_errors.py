import sys
import importlib
from unittest.mock import MagicMock

import pytest
import requests
from cryptography.fernet import Fernet

from shared import config


def _reload_auth(monkeypatch, key: bytes | None):
    if key is None:
        monkeypatch.delenv("IOL_TOKENS_KEY", raising=False)
        config.settings.tokens_key = None
    else:
        monkeypatch.setenv("IOL_TOKENS_KEY", key.decode())
        config.settings.tokens_key = key.decode()
    import infrastructure.iol.auth as auth_module
    importlib.reload(auth_module)
    return auth_module


@pytest.mark.skipif(sys.platform == "win32", reason="chmod not supported on Windows")
def test_save_tokens_chmod_failure(monkeypatch, tmp_path):
    key = Fernet.generate_key()
    auth_module = _reload_auth(monkeypatch, key)
    auth = auth_module.IOLAuth("u", "p", tokens_file=tmp_path / "t.json")

    def boom(*args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr(auth_module.os, "chmod", boom)
    with pytest.raises(RuntimeError):
        auth._save_tokens({"access_token": "x"})


def test_load_tokens_invalid_token(monkeypatch, tmp_path):
    key = Fernet.generate_key()
    auth_module = _reload_auth(monkeypatch, key)
    path = tmp_path / "t.json"
    path.write_bytes(b"not-a-token")

    auth = auth_module.IOLAuth("u", "p", tokens_file=path)
    assert auth.tokens == {}


def test_load_tokens_corrupted_json(monkeypatch, tmp_path):
    key = Fernet.generate_key()
    auth_module = _reload_auth(monkeypatch, key)
    path = tmp_path / "t.json"
    bad = Fernet(key).encrypt(b"{bad json")
    path.write_bytes(bad)

    auth = auth_module.IOLAuth("u", "p", tokens_file=path)
    assert auth.tokens == {}


@pytest.mark.parametrize("exc_cls", [requests.ConnectionError])
def test_login_network_errors(monkeypatch, tmp_path, exc_cls):
    key = Fernet.generate_key()
    auth_module = _reload_auth(monkeypatch, key)
    monkeypatch.setattr(
        auth_module.requests.Session,
        "post",
        MagicMock(side_effect=exc_cls("boom")),
    )
    auth = auth_module.IOLAuth("u", "p", tokens_file=tmp_path / "t.json")
    with pytest.raises(auth_module.NetworkError):
        auth.login()


def test_login_timeout_error(monkeypatch, tmp_path):
    key = Fernet.generate_key()
    auth_module = _reload_auth(monkeypatch, key)
    monkeypatch.setattr(
        auth_module.requests.Session,
        "post",
        MagicMock(side_effect=requests.Timeout("boom")),
    )
    auth = auth_module.IOLAuth("u", "p", tokens_file=tmp_path / "t.json")
    with pytest.raises(auth_module.TimeoutError):
        auth.login()


def test_login_request_exception(monkeypatch, tmp_path):
    key = Fernet.generate_key()
    auth_module = _reload_auth(monkeypatch, key)
    monkeypatch.setattr(
        auth_module.requests.Session,
        "post",
        MagicMock(side_effect=requests.RequestException("boom")),
    )
    auth = auth_module.IOLAuth("u", "p", tokens_file=tmp_path / "t.json")
    with pytest.raises(auth_module.NetworkError):
        auth.login()


def test_login_invalid_credentials(monkeypatch, tmp_path):
    key = Fernet.generate_key()
    auth_module = _reload_auth(monkeypatch, key)
    resp = MagicMock()
    resp.status_code = 401
    resp.text = "bad"
    monkeypatch.setattr(auth_module.requests.Session, "post", MagicMock(return_value=resp))
    auth = auth_module.IOLAuth("u", "p", tokens_file=tmp_path / "t.json")
    with pytest.raises(auth_module.InvalidCredentialsError):
        auth.login()


@pytest.mark.parametrize("exc_cls", [requests.ConnectionError])
def test_refresh_network_errors(monkeypatch, tmp_path, exc_cls):
    key = Fernet.generate_key()
    auth_module = _reload_auth(monkeypatch, key)
    monkeypatch.setattr(
        auth_module.requests.Session,
        "post",
        MagicMock(side_effect=exc_cls("boom")),
    )
    auth = auth_module.IOLAuth("u", "p", tokens_file=tmp_path / "t.json")
    auth.tokens = {"refresh_token": "r"}
    with pytest.raises(auth_module.NetworkError):
        auth.refresh()


def test_refresh_timeout_error(monkeypatch, tmp_path):
    key = Fernet.generate_key()
    auth_module = _reload_auth(monkeypatch, key)
    monkeypatch.setattr(
        auth_module.requests.Session,
        "post",
        MagicMock(side_effect=requests.Timeout("boom")),
    )
    auth = auth_module.IOLAuth("u", "p", tokens_file=tmp_path / "t.json")
    auth.tokens = {"refresh_token": "r"}
    with pytest.raises(auth_module.TimeoutError):
        auth.refresh()


def test_refresh_request_exception(monkeypatch, tmp_path):
    key = Fernet.generate_key()
    auth_module = _reload_auth(monkeypatch, key)
    monkeypatch.setattr(
        auth_module.requests.Session,
        "post",
        MagicMock(side_effect=requests.RequestException("boom")),
    )
    auth = auth_module.IOLAuth("u", "p", tokens_file=tmp_path / "t.json")
    auth.tokens = {"refresh_token": "r"}
    with pytest.raises(auth_module.NetworkError):
        auth.refresh()


def test_refresh_invalid_credentials(monkeypatch, tmp_path):
    key = Fernet.generate_key()
    auth_module = _reload_auth(monkeypatch, key)
    resp = MagicMock()
    resp.status_code = 401
    resp.text = "bad"
    monkeypatch.setattr(auth_module.requests.Session, "post", MagicMock(return_value=resp))
    auth = auth_module.IOLAuth("u", "p", tokens_file=tmp_path / "t.json")
    auth.tokens = {"refresh_token": "r"}
    with pytest.raises(auth_module.InvalidCredentialsError):
        auth.refresh()
    assert auth.tokens == {}


def test_clear_tokens_missing_file(monkeypatch, tmp_path):
    key = Fernet.generate_key()
    auth_module = _reload_auth(monkeypatch, key)
    path = tmp_path / "t.json"
    auth = auth_module.IOLAuth("u", "p", tokens_file=path)
    auth.tokens = {"a": 1}
    auth.clear_tokens()
    assert auth.tokens == {}
    assert not path.exists()