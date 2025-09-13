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
    st.container = MagicMock()
    st.rerun = MagicMock()
    return st


def test_login_page_rendered_when_missing_credentials(monkeypatch):
    monkeypatch.setenv("IOL_ALLOW_PLAIN_TOKENS", "1")
    sys.modules.pop("app", None)
    sys.modules.pop("shared.config", None)
    st = _make_streamlit()
    sys.modules["streamlit"] = st
    st.session_state.clear()

    app = importlib.import_module("app")

    login_mock = MagicMock()
    monkeypatch.setattr(app, "render_login_page", login_mock)
    monkeypatch.setattr(app, "get_fx_rates_cached", MagicMock(return_value=({}, None)))

    with pytest.raises(RuntimeError):
        app.main()

    login_mock.assert_called_once()
    st.stop.assert_called_once()


def test_successful_login_redirects_to_main_page(monkeypatch):
    monkeypatch.setenv("IOL_ALLOW_PLAIN_TOKENS", "1")
    sys.modules.pop("app", None)
    sys.modules.pop("ui.login", None)
    sys.modules.pop("shared.config", None)
    st = _make_streamlit()
    sys.modules["streamlit"] = st
    st.session_state.clear()

    from contextlib import nullcontext

    st.form = MagicMock(return_value=nullcontext())
    st.text_input = MagicMock()
    st.form_submit_button = MagicMock(return_value=True)

    login_mod = importlib.import_module("ui.login")
    provider = MagicMock()
    provider.login = MagicMock()
    monkeypatch.setattr(login_mod, "get_auth_provider", lambda: provider)

    st.session_state.update({"IOL_USERNAME": "u", "some_password": "p"})
    st.rerun.side_effect = RuntimeError("rerun")

    with pytest.raises(RuntimeError):
        login_mod.render_login_page()

    assert st.session_state.get("authenticated") is True
    assert not any("password" in k.lower() for k in st.session_state)
    assert "force_login" not in st.session_state

    st.rerun = MagicMock()
    sys.modules.pop("app", None)
    app = importlib.import_module("app")
    monkeypatch.setattr(app, "get_fx_rates_cached", MagicMock(return_value=({}, None)))
    monkeypatch.setattr(app, "render_header", MagicMock())
    monkeypatch.setattr(app, "render_action_menu", MagicMock())
    monkeypatch.setattr(app, "render_footer", MagicMock())
    monkeypatch.setattr(app, "build_iol_client", MagicMock())
    monkeypatch.setattr(app, "render_portfolio_section", MagicMock(return_value=None))
    login_mock = MagicMock()
    monkeypatch.setattr(app, "render_login_page", login_mock)

    app.main()

    login_mock.assert_not_called()
    app.render_header.assert_called_once()


def test_refresh_secs_triggers_rerun(monkeypatch):
    monkeypatch.setenv("IOL_ALLOW_PLAIN_TOKENS", "1")
    sys.modules.pop("app", None)
    sys.modules.pop("shared.config", None)
    st = _make_streamlit()
    st.cache_resource = MagicMock()
    st.session_state.clear()
    st.session_state.update({"IOL_USERNAME": "u", "authenticated": True, "last_refresh": 0})
    st.stop = MagicMock()  # no exception
    sys.modules["streamlit"] = st

    app = importlib.import_module("app")

    monkeypatch.setattr(app, "get_fx_rates_cached", MagicMock(return_value=({}, None)))
    monkeypatch.setattr(app, "render_header", MagicMock())
    monkeypatch.setattr(app, "render_action_menu", MagicMock())
    monkeypatch.setattr(app, "render_footer", MagicMock())
    monkeypatch.setattr(app, "build_iol_client", MagicMock())
    monkeypatch.setattr(app, "render_portfolio_section", MagicMock(return_value="1"))

    before = st.session_state["last_refresh"]
    app.main()
    assert st.rerun.called
    assert st.session_state["last_refresh"] >= before

def test_missing_tokens_file_forces_login(monkeypatch, tmp_path):
    monkeypatch.setenv("IOL_ALLOW_PLAIN_TOKENS", "1")
    sys.modules.pop("app", None)
    sys.modules.pop("shared.config", None)
    st = _make_streamlit()
    st.cache_resource = MagicMock()
    st.session_state.clear()
    st.session_state.update({"IOL_USERNAME": "u", "authenticated": True})
    sys.modules["streamlit"] = st
    export_stub = types.ModuleType("shared.export")
    export_stub.df_to_csv_bytes = lambda df: b""
    export_stub.fig_to_png_bytes = lambda fig: b""
    sys.modules["shared.export"] = export_stub

    token_file = tmp_path / "tokens" / "u-1234.json"
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text("{}", encoding="utf-8")
    token_file.unlink()

    app = importlib.import_module("app")

    login_mock = MagicMock()
    monkeypatch.setattr(app, "render_login_page", login_mock)
    monkeypatch.setattr(app, "get_fx_rates_cached", MagicMock(return_value=({}, None)))
    monkeypatch.setattr(app, "render_header", MagicMock())
    monkeypatch.setattr(app, "render_action_menu", MagicMock())
    monkeypatch.setattr(app, "render_footer", MagicMock())
    monkeypatch.setattr(app, "render_portfolio_section", MagicMock(return_value=None))

    def dummy_build():
        if not token_file.exists():
            st.session_state["force_login"] = True
            st.rerun()
        return None

    monkeypatch.setattr(app, "build_iol_client", dummy_build)

    st.rerun.side_effect = RuntimeError("rerun")

    with pytest.raises(RuntimeError):
        app.main()

    st.rerun.assert_called_once()

    with pytest.raises(RuntimeError):
        app.main()

    login_mock.assert_called_once()
