import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest


def _make_streamlit():
    st = types.ModuleType("streamlit")
    st.session_state = {}
    st.set_page_config = MagicMock()
    st.markdown = MagicMock()
    st.stop = MagicMock(side_effect=RuntimeError("stop"))
    st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
    st.caption = MagicMock()
    st.warning = MagicMock()
    st.error = MagicMock()
    st.container = MagicMock()
    st.rerun = MagicMock()
    st.cache_resource = lambda func=None, **kwargs: func
    st.cache_data = st.cache_resource
    st.title = MagicMock()
    return st


def test_login_logout_and_relogin(monkeypatch):
    monkeypatch.setenv("IOL_ALLOW_PLAIN_TOKENS", "1")
    sys.modules.pop("app", None)
    sys.modules.pop("ui.login", None)
    sys.modules.pop("shared.config", None)
    st = _make_streamlit()
    from contextlib import nullcontext

    st.form = MagicMock(return_value=nullcontext())
    st.text_input = MagicMock(side_effect=["u", "p", "u", "p"])
    st.form_submit_button = MagicMock(return_value=True)
    sys.modules["streamlit"] = st

    class DummyProvider:
        def __init__(self):
            self.logins = 0
            self.logouts = 0

        def login(self, user, password):
            self.logins += 1
            return {"access_token": "a"}

        def logout(self, user, password=""):
            self.logouts += 1
            for key in ("authenticated", "IOL_USERNAME", "session_id", "tokens"):
                st.session_state.pop(key, None)

        def build_client(self):  # pragma: no cover - required by interface
            return None, None

    provider = DummyProvider()

    from application import auth_service

    auth_service._provider = provider
    login_mod = importlib.import_module("ui.login")
    monkeypatch.setattr(login_mod, "get_auth_provider", lambda: provider)

    st.session_state.update({"some_password": "p"})
    st.rerun.side_effect = RuntimeError("rerun")
    with pytest.raises(RuntimeError):
        login_mod.render_login_page()
    st.session_state.pop("login_error", None)
    assert not any("password" in k.lower() for k in st.session_state)

    st.rerun = MagicMock()
    sys.modules.pop("app", None)
    app = importlib.import_module("app")
    monkeypatch.setattr(app, "get_fx_rates_cached", MagicMock(return_value=({}, None)))
    monkeypatch.setattr(app, "render_header", MagicMock())
    monkeypatch.setattr(app, "render_action_menu", MagicMock())
    monkeypatch.setattr(app, "render_footer", MagicMock())
    portfolio_mock = MagicMock(return_value=None)
    monkeypatch.setattr(app, "render_portfolio_section", portfolio_mock)
    monkeypatch.setattr(app, "build_iol_client", MagicMock())

    app.main()

    assert portfolio_mock.call_count == 1
    assert st.session_state.get("authenticated") is True
    assert "session_id" in st.session_state

    st.rerun = MagicMock()
    monkeypatch.setattr(auth_service, "st", st)
    auth_service.logout("u")

    assert st.rerun.called
    for key in ("session_id", "authenticated", "tokens"):
        assert key not in st.session_state

    st.session_state.update({"some_password": "p"})
    st.rerun.side_effect = RuntimeError("rerun")
    with pytest.raises(RuntimeError):
        login_mod.render_login_page()
    assert not any("password" in k.lower() for k in st.session_state)

    st.rerun = MagicMock()
    app.main()

    assert portfolio_mock.call_count == 2
