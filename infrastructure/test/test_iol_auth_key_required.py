import importlib
import pytest


def test_iol_auth_fails_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("IOL_TOKENS_KEY", raising=False)
    from shared import config
    config.settings.tokens_key = None
    import infrastructure.iol.auth as auth_module
    importlib.reload(auth_module)
    with pytest.raises(RuntimeError):
        auth_module.IOLAuth("u", "p", tokens_file=tmp_path / "t.json")
