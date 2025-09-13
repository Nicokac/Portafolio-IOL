import requests
import pytest
import sys
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

# Stub cryptography to avoid heavy dependency during tests
crypto_mod = types.ModuleType("cryptography")
fernet_mod = types.ModuleType("fernet")
setattr(fernet_mod, "Fernet", object)
setattr(fernet_mod, "InvalidToken", Exception)
crypto_mod.fernet = fernet_mod
sys.modules.setdefault("cryptography", crypto_mod)
sys.modules.setdefault("cryptography.fernet", fernet_mod)

from application.ta_service import fetch_with_indicators
from services import cache
from controllers import auth
from infrastructure.iol.auth import InvalidCredentialsError


def test_fetch_with_indicators_handles_yfinance_failure(monkeypatch):
    fetch_with_indicators.clear()

    def boom(*args, **kwargs):  # simulate network failure
        raise RuntimeError("fail")

    monkeypatch.setattr("application.ta_service.yf.download", boom)
    with pytest.raises(RuntimeError):
        fetch_with_indicators("AAPL")


def test_fetch_fx_rates_handles_network_error(monkeypatch):
    cache.fetch_fx_rates.clear()

    class FailProv:
        def get_rates(self):
            raise requests.RequestException("boom")

    monkeypatch.setattr(cache, "get_fx_provider", lambda: FailProv())
    data, error = cache.fetch_fx_rates()
    assert data == {}
    assert error is not None


def test_build_iol_client_handles_network_error(monkeypatch):
    mock_st = SimpleNamespace(session_state={"IOL_USERNAME": "u", "IOL_PASSWORD": "p"})
    monkeypatch.setattr(cache, "st", mock_st)

    class DummyAuth:
        def __init__(self, *a, **k):
            self.tokens = {"access_token": "x", "refresh_token": "r"}

        def refresh(self):
            return self.tokens

    monkeypatch.setattr(cache, "IOLAuth", DummyAuth)

    def fail_build(*args, **kwargs):
        raise requests.RequestException("net down")

    monkeypatch.setattr(cache, "_build_iol_client", fail_build)

    cli, err = cache.build_iol_client()
    assert cli is None
    assert "net down" in str(err)


def test_auth_controller_handles_network_error(monkeypatch):
    mock_st = SimpleNamespace(session_state={}, rerun=MagicMock())
    monkeypatch.setattr(auth, "st", mock_st)

    class DummyProvider:
        def build_client(self):
            return None, RuntimeError("net fail")

    monkeypatch.setattr(auth, "get_auth_provider", lambda: DummyProvider())

    cli = auth.build_iol_client()
    assert cli is None
    assert mock_st.session_state["login_error"] == "Error de conexión"
    assert mock_st.session_state["force_login"] is True
    assert "IOL_PASSWORD" not in mock_st.session_state
    mock_st.rerun.assert_called_once()


def test_auth_controller_handles_invalid_credentials(monkeypatch):
    mock_st = SimpleNamespace(session_state={}, rerun=MagicMock())
    monkeypatch.setattr(auth, "st", mock_st)

    class DummyProvider:
        def build_client(self):
            return None, InvalidCredentialsError()

    monkeypatch.setattr(auth, "get_auth_provider", lambda: DummyProvider())

    cli = auth.build_iol_client()
    assert cli is None
    assert mock_st.session_state["login_error"] == "Credenciales inválidas"
    assert mock_st.session_state["force_login"] is True
    assert "IOL_PASSWORD" not in mock_st.session_state
    mock_st.rerun.assert_called_once()
