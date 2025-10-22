import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest

from tests.fixtures.common import DummyCtx


def _make_streamlit():
    st = types.ModuleType("streamlit")
    st.session_state = {}
    st.set_page_config = MagicMock()
    st.markdown = MagicMock()
    st.stop = MagicMock(side_effect=RuntimeError("stop"))
    st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
    st.caption = MagicMock()
    st.warning = MagicMock()
    st.container = MagicMock()
    st.cache_resource = MagicMock()
    st.cache_data = MagicMock()
    st.rerun = MagicMock()
    st.form = MagicMock(return_value=DummyCtx())
    st.text_input = MagicMock(return_value="p")
    st.form_submit_button = MagicMock(return_value=True)
    st.error = MagicMock()
    return st


def test_valid_login_rerun_accesses_main_page(monkeypatch):
    for mod in ("streamlit", "ui.login", "app", "shared.config", "ui.ui_settings"):
        sys.modules.pop(mod, None)
    st = _make_streamlit()
    sys.modules["streamlit"] = st
    export_stub = types.ModuleType("shared.export")
    export_stub.df_to_csv_bytes = lambda df: b""
    export_stub.fig_to_png_bytes = lambda fig: b""
    sys.modules["shared.export"] = export_stub

    login_mod = importlib.import_module("ui.login")
    monkeypatch.setattr(login_mod.settings, "tokens_key", "k")
    monkeypatch.setattr(login_mod, "render_header", lambda: None)
    st.session_state.update({"IOL_USERNAME": "u"})

    class DummyProvider:
        def login(self, u, p):
            pass

    monkeypatch.setattr(login_mod, "get_auth_provider", lambda: DummyProvider())
    st.rerun.side_effect = RuntimeError("rerun")

    with pytest.raises(RuntimeError):
        login_mod.render_login_page()

    assert st.rerun.called
    assert st.session_state.get("authenticated") is True

    st.rerun = MagicMock()
    for mod in ("app", "shared.config", "ui.ui_settings"):
        sys.modules.pop(mod, None)
    app = importlib.import_module("app")
    monkeypatch.setattr(app, "get_fx_rates_cached", MagicMock(return_value=({}, None)))
    monkeypatch.setattr(app, "render_header", MagicMock())
    monkeypatch.setattr(app, "render_action_menu", MagicMock())
    monkeypatch.setattr(app, "render_footer", MagicMock())
    monkeypatch.setattr(app, "build_iol_client", MagicMock())
    monkeypatch.setattr(app, "render_portfolio_section", MagicMock(return_value=None))
    login_mock = MagicMock()
    monkeypatch.setattr(app, "render_login_page", login_mock)
    monkeypatch.setattr(app, "configure_logging", lambda **k: None)
    monkeypatch.setattr(app, "ensure_tokens_key", lambda: None)

    app.main([])

    login_mock.assert_not_called()
    st.stop.assert_not_called()
    assert st.session_state.get("authenticated") is True


def test_invalid_login_rerun_stays_on_login(monkeypatch):
    for mod in ("streamlit", "ui.login", "shared.config", "ui.ui_settings"):
        sys.modules.pop(mod, None)
    st = _make_streamlit()
    sys.modules["streamlit"] = st
    export_stub = types.ModuleType("shared.export")
    export_stub.df_to_csv_bytes = lambda df: b""
    export_stub.fig_to_png_bytes = lambda fig: b""
    sys.modules["shared.export"] = export_stub

    login_mod = importlib.import_module("ui.login")
    monkeypatch.setattr(login_mod.settings, "tokens_key", "k")
    monkeypatch.setattr(login_mod, "render_header", lambda: None)
    st.session_state.update({"IOL_USERNAME": "u"})

    class DummyProvider:
        def login(self, u, p):
            raise login_mod.InvalidCredentialsError()

    monkeypatch.setattr(login_mod, "get_auth_provider", lambda: DummyProvider())
    st.rerun.side_effect = RuntimeError("rerun")

    with pytest.raises(RuntimeError):
        login_mod.render_login_page()

    assert st.rerun.called
    assert st.session_state.get("login_error") == "Usuario o contraseña inválidos"
    assert "authenticated" not in st.session_state

    st.rerun = MagicMock()
    st.form_submit_button = MagicMock(return_value=False)
    captured = {}
    st.error = lambda msg: captured.setdefault("msg", msg)

    login_mod.render_login_page()

    assert captured.get("msg") == "Usuario o contraseña inválidos"


def test_expired_session_forces_login(monkeypatch):
    for mod in (
        "streamlit",
        "controllers.auth",
        "app",
        "shared.config",
        "ui.ui_settings",
    ):
        sys.modules.pop(mod, None)
    st = _make_streamlit()
    sys.modules["streamlit"] = st
    export_stub = types.ModuleType("shared.export")
    export_stub.df_to_csv_bytes = lambda df: b""
    export_stub.fig_to_png_bytes = lambda fig: b""
    sys.modules["shared.export"] = export_stub
    st.session_state.update({"authenticated": True})

    auth_mod = importlib.import_module("controllers.auth")

    class DummyProvider:
        def build_client(self):
            return None, RuntimeError("expired")

    monkeypatch.setattr(auth_mod, "get_auth_provider", lambda: DummyProvider())
    st.rerun.side_effect = RuntimeError("rerun")

    with pytest.raises(RuntimeError):
        auth_mod.build_iol_client()

    assert st.rerun.called
    assert st.session_state.get("force_login") is True
    assert st.session_state.get("login_error") == "Error de conexión"

    st.rerun = MagicMock()
    for mod in ("app", "shared.config", "ui.ui_settings"):
        sys.modules.pop(mod, None)
    app = importlib.import_module("app")
    login_mock = MagicMock()
    monkeypatch.setattr(app, "render_login_page", login_mock)
    monkeypatch.setattr(app, "get_fx_rates_cached", MagicMock(return_value=({}, None)))
    monkeypatch.setattr(app, "render_header", MagicMock())
    monkeypatch.setattr(app, "render_action_menu", MagicMock())
    monkeypatch.setattr(app, "render_footer", MagicMock())
    monkeypatch.setattr(app, "build_iol_client", MagicMock())
    monkeypatch.setattr(app, "render_portfolio_section", MagicMock(return_value=None))
    monkeypatch.setattr(app, "configure_logging", lambda **k: None)
    monkeypatch.setattr(app, "ensure_tokens_key", lambda: None)

    with pytest.raises(RuntimeError):
        app.main([])

    login_mock.assert_called_once()
    st.stop.assert_called_once()
