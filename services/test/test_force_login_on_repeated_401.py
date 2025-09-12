import sys
import types
from types import SimpleNamespace
import requests

# Stub cryptography to avoid heavy dependency
crypto_mod = types.ModuleType("cryptography")
fernet_mod = types.ModuleType("fernet")
class DummyFernet:
    def __init__(self, key: bytes):
        pass
    def encrypt(self, data: bytes) -> bytes:
        return data
    def decrypt(self, data: bytes) -> bytes:
        return data
setattr(fernet_mod, "Fernet", DummyFernet)
setattr(fernet_mod, "InvalidToken", Exception)
crypto_mod.fernet = fernet_mod
sys.modules.setdefault("cryptography", crypto_mod)
sys.modules.setdefault("cryptography.fernet", fernet_mod)

from services import cache as svc_cache
from infrastructure.iol.legacy.iol_client import IOLClient
from infrastructure.iol import auth as auth_mod
from shared import config


def test_repeated_401_forces_login(monkeypatch):
    config.settings.tokens_key = "k"
    auth_mod.FERNET = DummyFernet(b"k")
    monkeypatch.setattr(IOLClient, "_ensure_market_auth", lambda self: None)
    cli = IOLClient("u", "p", tokens_file="/tmp/tokens.json")
    monkeypatch.setattr(cli.auth, "auth_header", lambda: {"Authorization": "Bearer x"})

    refreshed = {"called": 0}

    def refresh():
        refreshed["called"] += 1

    monkeypatch.setattr(cli.auth, "refresh", refresh)

    class Resp:
        status_code = 401
        text = "Unauthorized"
        def raise_for_status(self):
            raise requests.HTTPError(response=self)

    monkeypatch.setattr(cli.session, "request", lambda *a, **k: Resp())

    monkeypatch.setattr(svc_cache, "st", SimpleNamespace(session_state={}))
    svc_cache.fetch_portfolio.clear()
    payload = svc_cache.fetch_portfolio(cli)

    assert svc_cache.st.session_state["force_login"] is True
    assert refreshed["called"] == 1
    assert payload == {"_cached": True}

