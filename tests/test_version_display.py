from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.version import __version__
from ui.login import render_login_page
import app as main_app
import ui.footer
from unittest.mock import MagicMock


class DummyCtx:
    def __enter__(self):
        return None
    def __exit__(self, *exc):
        return False


def test_version_shown_in_login(monkeypatch):
    monkeypatch.setattr("ui.login.settings.tokens_key", "dummy")
    monkeypatch.setattr("ui.login.render_header", lambda *a, **k: None)
    monkeypatch.setattr(ui.footer, "get_version", lambda: (__version__, "hoy"))
    mock_markdown = MagicMock()
    monkeypatch.setattr(ui.footer.st, "markdown", mock_markdown)
    monkeypatch.setattr("ui.login.st.warning", lambda *a, **k: None)
    monkeypatch.setattr("ui.login.st.error", lambda *a, **k: None)
    monkeypatch.setattr("ui.login.st.text_input", lambda *a, **k: "")
    monkeypatch.setattr("ui.login.st.form_submit_button", lambda *a, **k: False)
    monkeypatch.setattr("ui.login.st.form", lambda *a, **k: DummyCtx())
    render_login_page()
    assert any(__version__ in str(call.args[0]) for call in mock_markdown.call_args_list)


def test_version_shown_in_main_app(monkeypatch):
    monkeypatch.setattr(main_app, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(main_app, "ensure_tokens_key", lambda: None)
    monkeypatch.setattr(main_app, "get_fx_rates_cached", lambda: ({}, None))
    monkeypatch.setattr(main_app, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(main_app, "render_action_menu", lambda *a, **k: None)
    monkeypatch.setattr(main_app, "build_iol_client", lambda: None)
    monkeypatch.setattr(main_app, "render_portfolio_section", lambda *a, **k: None)
    monkeypatch.setattr(ui.footer, "get_version", lambda: (__version__, "hoy"))
    mock_markdown = MagicMock()
    monkeypatch.setattr(ui.footer.st, "markdown", mock_markdown)
    monkeypatch.setattr(main_app.st, "session_state", {"authenticated": True})
    monkeypatch.setattr(main_app.st, "stop", lambda: None)
    monkeypatch.setattr(main_app.st, "columns", lambda *a, **k: (DummyCtx(), DummyCtx()))
    monkeypatch.setattr(main_app.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(main_app.st, "container", lambda: DummyCtx())
    main_app.main([])
    assert any(__version__ in str(call.args[0]) for call in mock_markdown.call_args_list)
