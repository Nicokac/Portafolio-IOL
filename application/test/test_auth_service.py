import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
import streamlit as st

from application import auth_service
from application.auth_service import AuthenticationError


def test_login_success(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    with patch("application.auth_service.IOLAuth") as mock_auth:
        mock_auth.return_value.login.return_value = {"access_token": "tok"}
        user = "user"
        tokens = auth_service.login(user, "pass")
        assert tokens["access_token"] == "tok"
        assert "client_salt" in st.session_state
        assert len(st.session_state["client_salt"]) == 32
        user_hash = hashlib.sha256(user.encode()).hexdigest()[:12]
        expected = Path("tokens") / f"{user}-{user_hash}.json"
        mock_auth.assert_called_once_with(
            user,
            "pass",
            tokens_file=expected,
            allow_plain_tokens=False,
        )
        mock_auth.return_value.login.assert_called_once()


def test_login_invalid_raises(monkeypatch):
    monkeypatch.setattr(st, "session_state", {})
    with patch("application.auth_service.IOLAuth") as mock_auth:
        mock_auth.return_value.login.return_value = {}
        user = "u"
        with pytest.raises(AuthenticationError):
            auth_service.login(user, "p")
        assert "client_salt" not in st.session_state
        user_hash = hashlib.sha256(user.encode()).hexdigest()[:12]
        expected = Path("tokens") / f"{user}-{user_hash}.json"
        mock_auth.assert_called_once_with(
            user,
            "p",
            tokens_file=expected,
            allow_plain_tokens=False,
        )


def test_logout_clears_session_state(monkeypatch):
    monkeypatch.setattr(
        st,
        "session_state",
        {
            "session_id": "A",
            "IOL_USERNAME": "user",
            "authenticated": True,
            "client_salt": "s",
            "tokens_file": "foo",
            "x": 1,
        },
    )
    monkeypatch.setattr(st, "rerun", lambda *a, **k: None)
    with patch("application.auth_service.IOLAuth") as mock_auth:
        user = "user"
        auth_service.logout(user)
        user_hash = hashlib.sha256(user.encode()).hexdigest()[:12]
        expected = Path("tokens") / f"{user}-{user_hash}.json"
        mock_auth.assert_called_once_with(
            "user",
            "",
            tokens_file=expected,
            allow_plain_tokens=False,
        )
        mock_auth.return_value.clear_tokens.assert_called_once()
    assert st.session_state.get("force_login") is True
    assert st.session_state.get("logout_done") is True


def test_logout_generates_new_session_id(monkeypatch):
    from types import SimpleNamespace

    import app

    monkeypatch.setattr(st, "session_state", {"session_id": "old"})
    monkeypatch.setattr(st, "rerun", lambda *a, **k: None)
    with patch("application.auth_service.IOLAuth"):
        auth_service.logout("user")
    assert st.session_state.get("force_login") is True
    assert st.session_state.get("logout_done") is True

    monkeypatch.setattr(app, "render_login_page", lambda: None)
    monkeypatch.setattr(app, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(app, "render_action_menu", lambda: None)
    monkeypatch.setattr(app, "render_footer", lambda: None)
    monkeypatch.setattr(app, "render_portfolio_section", lambda *a, **k: None)
    monkeypatch.setattr(app, "get_fx_rates_cached", lambda: (None, None))
    monkeypatch.setattr(app, "build_iol_client", lambda: object())
    monkeypatch.setattr(app, "configure_logging", lambda level=None, json_format=None: None)
    monkeypatch.setattr(app, "ensure_tokens_key", lambda: None)
    monkeypatch.setattr(
        app,
        "_parse_args",
        lambda argv: SimpleNamespace(log_level=None, log_format=None),
    )
    monkeypatch.setattr(app, "uuid4", lambda: SimpleNamespace(hex="new_session"))
    monkeypatch.setattr(st, "stop", lambda: (_ for _ in ()).throw(SystemExit))

    with pytest.raises(SystemExit):
        app.main([])

    assert st.session_state.get("session_id") == "new_session"


def test_logout_clears_cached_queries(monkeypatch):
    import streamlit as st

    from application import auth_service
    from services import cache as svc_cache

    # Compartir session_state entre módulos y evitar rerun
    monkeypatch.setattr(st, "session_state", {"IOL_USERNAME": "user"})
    monkeypatch.setattr(st, "rerun", lambda *a, **k: None)

    # Evitar operaciones reales de autenticación
    class DummyAuth:
        def __init__(self, *a, **k):
            pass

        def clear_tokens(self):
            pass

    monkeypatch.setattr(auth_service, "IOLAuth", DummyAuth)

    # Dummy cache resource used by logout
    class DummyGetClientCached:
        def clear(self, key=None):
            pass

    monkeypatch.setattr(svc_cache, "get_client_cached", DummyGetClientCached())

    # Proveedores y clientes simulados para las funciones cacheadas
    class DummyClient:
        def __init__(self):
            self.portfolio_calls = 0
            self.quotes_calls = 0

        def get_portfolio(self, country="argentina"):
            self.portfolio_calls += 1
            return {"n": self.portfolio_calls}

        def get_quotes_bulk(self, items):
            self.quotes_calls += 1
            return {("m", "s"): {"last": self.quotes_calls, "chg_pct": None}}

    class DummyFXProvider:
        def __init__(self):
            self.calls = 0

        def get_rates(self):
            self.calls += 1
            return {"USD": self.calls}, None

        def close(self):
            pass

    provider = DummyFXProvider()
    monkeypatch.setattr(svc_cache, "get_fx_provider", lambda: provider)

    # Limpiar caches iniciales
    svc_cache.fetch_portfolio.clear()
    svc_cache.fetch_quotes_bulk.clear()
    svc_cache.fetch_fx_rates.clear()

    cli = DummyClient()
    items = [("m", "s")]

    # Primeras llamadas pobladas en cache
    assert svc_cache.fetch_portfolio(cli) == {"n": 1}
    assert svc_cache.fetch_portfolio(cli) == {"n": 1}
    assert cli.portfolio_calls == 1

    first_quote = svc_cache.fetch_quotes_bulk(cli, items)
    assert set(first_quote.keys()) == {("m", "s")}
    assert first_quote[("m", "s")]["last"] == 1
    assert first_quote[("m", "s")]["chg_pct"] is None
    second_quote = svc_cache.fetch_quotes_bulk(cli, items)
    assert second_quote[("m", "s")]["last"] == 1
    assert second_quote[("m", "s")]["chg_pct"] is None
    assert cli.quotes_calls == 1

    assert svc_cache.fetch_fx_rates() == ({"USD": 1}, None)
    assert svc_cache.fetch_fx_rates() == ({"USD": 1}, None)
    assert provider.calls == 1

    # Logout debe limpiar caches
    auth_service.logout("user")

    assert svc_cache.fetch_portfolio(cli) == {"n": 2}
    assert cli.portfolio_calls == 2

    refreshed_quote = svc_cache.fetch_quotes_bulk(cli, items)
    assert refreshed_quote[("m", "s")]["last"] == 2
    assert refreshed_quote[("m", "s")]["chg_pct"] is None
    assert cli.quotes_calls == 2

    assert svc_cache.fetch_fx_rates() == ({"USD": 2}, None)
    assert provider.calls == 2
