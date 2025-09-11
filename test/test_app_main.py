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
    sys.modules.pop("app", None)
    st = _make_streamlit()
    sys.modules["streamlit"] = st

    from shared import config
    monkeypatch.setattr(config.settings, "IOL_USERNAME", None)
    monkeypatch.setattr(config.settings, "IOL_PASSWORD", None)

    app = importlib.import_module("app")

    login_mock = MagicMock()
    monkeypatch.setattr(app, "render_login_page", login_mock)
    monkeypatch.setattr(app, "get_fx_rates_cached", MagicMock(return_value={}))

    with pytest.raises(RuntimeError):
        app.main()

    login_mock.assert_called_once()
    st.stop.assert_called_once()


def test_refresh_secs_triggers_rerun(monkeypatch):
    sys.modules.pop("app", None)
    st = _make_streamlit()
    st.session_state.update({"IOL_USERNAME": "u", "IOL_PASSWORD": "p", "last_refresh": 0})
    st.stop = MagicMock()  # no exception
    sys.modules["streamlit"] = st

    app = importlib.import_module("app")

    monkeypatch.setattr(app, "get_fx_rates_cached", MagicMock(return_value={}))
    monkeypatch.setattr(app, "render_header", MagicMock())
    monkeypatch.setattr(app, "render_action_menu", MagicMock())
    monkeypatch.setattr(app, "render_footer", MagicMock())
    monkeypatch.setattr(app, "build_iol_client", MagicMock())
    monkeypatch.setattr(app, "render_portfolio_section", MagicMock(return_value="1"))

    before = st.session_state["last_refresh"]
    app.main()
    assert st.rerun.called
    assert st.session_state["last_refresh"] >= before

