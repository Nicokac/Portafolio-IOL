import sys
import types
from types import SimpleNamespace

import pytest

from infrastructure.iol import auth as auth_mod
from services import cache as svc_cache
from shared import config

# Stub cryptography to avoid heavy dependency
crypto_mod = types.ModuleType("cryptography")
fernet_mod = types.ModuleType("fernet")


class DummyFernet:
    def __init__(self, key: bytes):
        pass

    def encrypt(self, data: bytes) -> bytes:  # pragma: no cover - simple stub
        return data

    def decrypt(self, data: bytes) -> bytes:  # pragma: no cover - simple stub
        return data


setattr(fernet_mod, "Fernet", DummyFernet)
setattr(fernet_mod, "InvalidToken", Exception)
crypto_mod.fernet = fernet_mod
sys.modules.setdefault("cryptography", crypto_mod)
sys.modules.setdefault("cryptography.fernet", fernet_mod)


def test_repeated_401_forces_login(monkeypatch):
    config.settings.tokens_key = "k"
    auth_mod.FERNET = DummyFernet(b"k")

    cleared = {"called": False}

    def clear_tokens():
        cleared["called"] = True

    class DummyCli:
        def __init__(self):
            self._cli = SimpleNamespace(auth=SimpleNamespace(clear_tokens=clear_tokens))

        def get_portfolio(self, country="argentina"):
            raise svc_cache.InvalidCredentialsError()

    cli = DummyCli()

    rerun_called = {"called": False}

    def rerun():
        rerun_called["called"] = True
        raise RuntimeError("rerun")

    monkeypatch.setattr(svc_cache, "st", SimpleNamespace(session_state={}, rerun=rerun))
    svc_cache.fetch_portfolio.clear()

    def dummy_logout(user="", password=""):
        svc_cache.st.session_state["force_login"] = True
        svc_cache.st.rerun()

    monkeypatch.setattr("application.auth_service.logout", dummy_logout)

    with pytest.raises(RuntimeError, match="rerun"):
        svc_cache.fetch_portfolio(cli)

    assert svc_cache.st.session_state["force_login"] is True
    assert cleared["called"] is True
    assert rerun_called["called"] is True
