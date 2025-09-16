from pathlib import Path
import re
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.version import __version__
from ui.login import render_login_page
import app as main_app
import ui.footer
from unittest.mock import MagicMock
from shared.time_provider import TimeProvider


class DummyCtx:
    def __enter__(self):
        return None
    def __exit__(self, *exc):
        return False


def setup_footer_mocks(monkeypatch):
    tz = TimeProvider.TIMEZONE
    fixed_dt = datetime(2024, 1, 2, 3, 4, 5, tzinfo=tz)
    formatted = fixed_dt.strftime(TimeProvider.DATETIME_FORMAT)
    call_counts = {"now": 0, "now_datetime": 0}

    def fake_now(cls):
        call_counts["now"] += 1
        return formatted

    def fake_now_datetime(cls):
        call_counts["now_datetime"] += 1
        return fixed_dt

    monkeypatch.setattr(TimeProvider, "now", classmethod(fake_now))
    monkeypatch.setattr(TimeProvider, "now_datetime", classmethod(fake_now_datetime))
    monkeypatch.setattr(ui.footer, "get_version", lambda: __version__)
    mock_markdown = MagicMock()
    monkeypatch.setattr(ui.footer.st, "markdown", mock_markdown)
    return mock_markdown, call_counts, fixed_dt, formatted


def test_version_shown_in_login(monkeypatch):
    monkeypatch.setattr("ui.login.settings.tokens_key", "dummy")
    monkeypatch.setattr("ui.login.render_header", lambda *a, **k: None)
    mock_markdown, call_counts, fixed_dt, formatted = setup_footer_mocks(monkeypatch)
    monkeypatch.setattr("ui.login.st.warning", lambda *a, **k: None)
    monkeypatch.setattr("ui.login.st.error", lambda *a, **k: None)
    monkeypatch.setattr("ui.login.st.text_input", lambda *a, **k: "")
    monkeypatch.setattr("ui.login.st.form_submit_button", lambda *a, **k: False)
    monkeypatch.setattr("ui.login.st.form", lambda *a, **k: DummyCtx())
    render_login_page()
    rendered_blocks = [str(call.args[0]) for call in mock_markdown.call_args_list]
    assert any(f"Versión {__version__}" in block for block in rendered_blocks)
    timestamp_pattern = re.compile(r"\b\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\b")
    assert any(timestamp_pattern.search(block) for block in rendered_blocks)
    assert any(formatted in block for block in rendered_blocks)
    assert call_counts["now"] >= 1
    assert call_counts["now_datetime"] >= 1
    assert getattr(TimeProvider.TIMEZONE, "key", "") == "America/Argentina/Buenos_Aires"
    assert fixed_dt.tzinfo is TimeProvider.TIMEZONE


def test_version_shown_in_main_app(monkeypatch):
    monkeypatch.setattr(main_app, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(main_app, "ensure_tokens_key", lambda: None)
    monkeypatch.setattr(main_app, "get_fx_rates_cached", lambda: ({}, None))
    monkeypatch.setattr(main_app, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(main_app, "render_action_menu", lambda *a, **k: None)
    monkeypatch.setattr(main_app, "build_iol_client", lambda: None)
    monkeypatch.setattr(main_app, "render_portfolio_section", lambda *a, **k: None)
    mock_markdown, call_counts, fixed_dt, formatted = setup_footer_mocks(monkeypatch)
    monkeypatch.setattr(main_app.st, "session_state", {"authenticated": True})
    monkeypatch.setattr(main_app.st, "stop", lambda: None)
    monkeypatch.setattr(main_app.st, "columns", lambda *a, **k: (DummyCtx(), DummyCtx()))
    monkeypatch.setattr(main_app.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(main_app.st, "container", lambda: DummyCtx())
    main_app.main([])
    rendered_blocks = [str(call.args[0]) for call in mock_markdown.call_args_list]
    assert any(f"Versión {__version__}" in block for block in rendered_blocks)
    timestamp_pattern = re.compile(r"\b\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\b")
    assert any(timestamp_pattern.search(block) for block in rendered_blocks)
    assert any(formatted in block for block in rendered_blocks)
    assert call_counts["now"] >= 1
    assert call_counts["now_datetime"] >= 1
    assert getattr(TimeProvider.TIMEZONE, "key", "") == "America/Argentina/Buenos_Aires"
    assert fixed_dt.tzinfo is TimeProvider.TIMEZONE
