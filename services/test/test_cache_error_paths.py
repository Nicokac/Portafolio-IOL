import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from services import cache as svc_cache

# --- _trigger_logout ---

def test_trigger_logout_exception(monkeypatch, caplog):
    state = {"IOL_USERNAME": "u"}
    monkeypatch.setattr(svc_cache, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr("application.auth_service.logout", MagicMock(side_effect=RuntimeError("boom")))
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError):
            svc_cache._trigger_logout()
    assert "auto logout failed" in caplog.text


# --- fetch_portfolio ---

def test_fetch_portfolio_handles_invalid_credentials(monkeypatch):
    state = {}
    monkeypatch.setattr(svc_cache, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr("shared.cache.st", SimpleNamespace(session_state=state))
    logout_mock = MagicMock()
    monkeypatch.setattr(svc_cache, "_trigger_logout", logout_mock)
    clear_tokens = MagicMock()

    class DummyCli:
        def __init__(self):
            self._cli = SimpleNamespace(auth=SimpleNamespace(clear_tokens=clear_tokens))

        def get_portfolio(self):
            raise svc_cache.InvalidCredentialsError()

    svc_cache.fetch_portfolio.clear()
    cli = DummyCli()
    result = svc_cache.fetch_portfolio(cli)
    assert result == {"_cached": True}
    logout_mock.assert_called_once()
    clear_tokens.assert_called_once()


def test_fetch_portfolio_handles_request_exception(monkeypatch):
    state = {}
    monkeypatch.setattr(svc_cache, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr("shared.cache.st", SimpleNamespace(session_state=state))
    logout_mock = MagicMock()
    monkeypatch.setattr(svc_cache, "_trigger_logout", logout_mock)

    class DummyCli:
        def get_portfolio(self):
            raise requests.RequestException("net")

    svc_cache.fetch_portfolio.clear()
    cli = DummyCli()
    result = svc_cache.fetch_portfolio(cli)
    assert result == {"_cached": True}
    logout_mock.assert_not_called()


# --- _get_quote_cached ---

def test_get_quote_cached_handles_invalid_credentials(monkeypatch):
    state = {}
    monkeypatch.setattr(svc_cache, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr("shared.cache.st", SimpleNamespace(session_state=state))
    clear_tokens = MagicMock()

    class DummyCli:
        def __init__(self):
            self._cli = SimpleNamespace(auth=SimpleNamespace(clear_tokens=clear_tokens))

        def get_quote(self, mercado, simbolo):
            raise svc_cache.InvalidCredentialsError()

    logout_mock = MagicMock()
    monkeypatch.setattr(svc_cache, "_trigger_logout", logout_mock)
    svc_cache._QUOTE_CACHE.clear()
    result = svc_cache._get_quote_cached(DummyCli(), "bcba", "GGAL")
    assert result == {"last": None, "chg_pct": None}
    logout_mock.assert_called_once()
    clear_tokens.assert_called_once()


# --- build_iol_client ---

def test_build_iol_client_missing_user(monkeypatch):
    state = {}
    monkeypatch.setattr(svc_cache, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr("shared.cache.st", SimpleNamespace(session_state=state))
    monkeypatch.setattr(svc_cache.settings, "IOL_USERNAME", None)
    cli, err = svc_cache.build_iol_client()
    assert cli is None
    assert isinstance(err, RuntimeError)
    assert str(err) == "missing user"


def test_build_iol_client_invalid_credentials(monkeypatch):
    state = {}
    monkeypatch.setattr(svc_cache, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr("shared.cache.st", SimpleNamespace(session_state=state))
    monkeypatch.setattr(svc_cache.settings, "IOL_USERNAME", None)
    monkeypatch.setattr(
        svc_cache,
        "get_client_cached",
        MagicMock(side_effect=svc_cache.InvalidCredentialsError("bad")),
    )
    logout_mock = MagicMock()
    monkeypatch.setattr(svc_cache, "_trigger_logout", logout_mock)
    cli, err = svc_cache.build_iol_client(user="u")
    assert cli is None
    assert isinstance(err, svc_cache.InvalidCredentialsError)
    logout_mock.assert_called_once()


def test_build_iol_client_generic_exception(monkeypatch, caplog):
    state = {}
    monkeypatch.setattr(svc_cache, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr("shared.cache.st", SimpleNamespace(session_state=state))
    monkeypatch.setattr(svc_cache.settings, "IOL_USERNAME", None)
    monkeypatch.setattr(
        svc_cache,
        "get_client_cached",
        MagicMock(side_effect=RuntimeError("boom")),
    )
    logout_mock = MagicMock()
    monkeypatch.setattr(svc_cache, "_trigger_logout", logout_mock)
    with caplog.at_level(logging.ERROR):
        cli, err = svc_cache.build_iol_client(user="u")
    assert cli is None
    assert isinstance(err, RuntimeError)
    logout_mock.assert_not_called()
    assert "build_iol_client failed" in caplog.text
