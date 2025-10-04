import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from infrastructure.iol.client import IOLClient

@pytest.mark.skipif(sys.platform.startswith("win"), reason="chmod not supported on Windows")
def test_iol_auth_login_and_clear_tokens(tmp_path, monkeypatch):
    tokens_path = tmp_path / "tokens.json"
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"access_token": "abc"}
    mock_resp.status_code = 200

    key = Fernet.generate_key()
    monkeypatch.setenv("IOL_TOKENS_KEY", key.decode())
    from shared import config
    config.settings.tokens_key = key.decode()

    with patch("requests.Session.post", return_value=mock_resp):
        import importlib
        import infrastructure.iol.auth as auth_module
        importlib.reload(auth_module)
        auth = auth_module.IOLAuth("user", "pass", tokens_file=tokens_path)
        tokens = auth.login()
        assert tokens["access_token"] == "abc"
        assert "timestamp" in tokens
        raw = tokens_path.read_bytes()
        assert raw != b'{"access_token": "abc"}'
        decrypted = Fernet(key).decrypt(raw).decode()
        stored = json.loads(decrypted)
        assert stored["access_token"] == "abc"
        assert "timestamp" in stored
        assert (tokens_path.stat().st_mode & 0o777) == 0o600
        auth.clear_tokens()
        assert not tokens_path.exists()

def test_iol_auth_expired_tokens_triggers_login(tmp_path, monkeypatch):
    tokens_path = tmp_path / "tokens.json"
    key = Fernet.generate_key()
    monkeypatch.setenv("IOL_TOKENS_KEY", key.decode())
    from shared import config
    config.settings.tokens_key = key.decode()
    # Write expired tokens
    expired = {"access_token": "old", "timestamp": 0}
    content = json.dumps(expired).encode("utf-8")
    content = Fernet(key).encrypt(content)
    tokens_path.write_bytes(content)

    import importlib
    import infrastructure.iol.auth as auth_module
    importlib.reload(auth_module)

    calls = {"count": 0}

    def fake_login(self):
        calls["count"] += 1
        self.tokens = {"access_token": "new"}
        self._save_tokens(self.tokens)
        return self.tokens

    monkeypatch.setattr(auth_module.IOLAuth, "login", fake_login)

    auth = auth_module.IOLAuth("user", "pass", tokens_file=tokens_path)
    assert auth.tokens["access_token"] == "new"
    assert calls["count"] == 1
    stored = json.loads(Fernet(key).decrypt(tokens_path.read_bytes()))
    assert stored["access_token"] == "new"
    assert "timestamp" in stored and stored["timestamp"] > 0


def test_iol_client_get_portfolio_uses_cache_on_failure(tmp_path, monkeypatch):
    cache_file = tmp_path / "portfolio.json"
    monkeypatch.setattr("infrastructure.iol.client.PORTFOLIO_CACHE", cache_file)
    monkeypatch.setattr(IOLClient, "_ensure_market_auth", lambda self: None, raising=False)

    cli = IOLClient("user", "pass", tokens_file=tmp_path / "tok.json")

    monkeypatch.setattr(cli, "_fetch_portfolio_live", lambda: {"activos": [1]}, raising=False)
    data1 = cli.get_portfolio()
    assert data1 == {"activos": [1]}
    assert json.loads(cache_file.read_text()) == {"activos": [1]}

    import requests

    def fail():
        raise requests.RequestException("boom")

    monkeypatch.setattr(cli, "_fetch_portfolio_live", fail, raising=False)
    data2 = cli.get_portfolio()
    assert data2 == {"activos": [1]}
