from pathlib import Path
import re
from datetime import datetime
import sys
from zoneinfo import ZoneInfo
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.version import __version__
from ui.login import render_login_page
import app as main_app
import ui.footer
from unittest.mock import MagicMock
from shared.time_provider import TimeProvider

from shared.time_provider import TIMEZONE, TimeSnapshot


class DummyCtx:
    def __enter__(self):
        return None
    def __exit__(self, *exc):
        return False


class FixedTimeProvider:
    def __init__(self, snapshot: TimeSnapshot):
        self._snapshot = snapshot
        self.calls = 0

    def now(self):
        self.calls += 1
        return self._snapshot


def setup_footer_mocks(monkeypatch):
    timezone = ZoneInfo(TIMEZONE)
    fixed_dt = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone)
    snapshot = TimeSnapshot(fixed_dt.strftime("%Y-%m-%d %H:%M:%S"), fixed_dt)
    provider_stub = FixedTimeProvider(snapshot)
    monkeypatch.setattr(ui.footer, "TimeProvider", provider_stub)
    monkeypatch.setattr(ui.footer, "get_version", lambda: __version__)
    mock_markdown = MagicMock()
    monkeypatch.setattr(ui.footer.st, "markdown", mock_markdown)
    return mock_markdown, provider_stub, snapshot

def test_version_shown_in_login(monkeypatch):
    monkeypatch.setattr("ui.login.settings.tokens_key", "dummy")
    monkeypatch.setattr("ui.login.render_header", lambda *a, **k: None)
    mock_markdown, time_provider_stub, snapshot = setup_footer_mocks(monkeypatch)
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
    expected_timestamp = snapshot.text
    assert any(expected_timestamp in block for block in rendered_blocks)
    assert time_provider_stub.calls == 1
    tzinfo = snapshot.moment.tzinfo
    assert isinstance(tzinfo, ZoneInfo)
    assert tzinfo.key == TIMEZONE


def test_version_shown_in_main_app(monkeypatch):
    monkeypatch.setattr(main_app, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(main_app, "ensure_tokens_key", lambda: None)
    monkeypatch.setattr(main_app, "get_fx_rates_cached", lambda: ({}, None))
    monkeypatch.setattr(main_app, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(main_app, "render_action_menu", lambda *a, **k: None)
    monkeypatch.setattr(main_app, "build_iol_client", lambda: None)
    monkeypatch.setattr(main_app, "render_portfolio_section", lambda *a, **k: None)
    mock_markdown, time_provider_stub, snapshot = setup_footer_mocks(monkeypatch)
    monkeypatch.setattr(main_app, "TimeProvider", time_provider_stub)
    monkeypatch.setattr(main_app.st, "session_state", {"authenticated": True})
    monkeypatch.setattr(main_app.st, "stop", lambda: None)
    monkeypatch.setattr(main_app.st, "columns", lambda *a, **k: (DummyCtx(), DummyCtx()))
    captions: list[str] = []

    def capture_caption(value, *args, **kwargs):
        captions.append(value)

    monkeypatch.setattr(main_app.st, "caption", capture_caption)
    monkeypatch.setattr(main_app.st, "container", lambda: DummyCtx())
    main_app.main([])
    rendered_blocks = [str(call.args[0]) for call in mock_markdown.call_args_list]
    assert any(f"Versión {__version__}" in block for block in rendered_blocks)
    timestamp_pattern = re.compile(r"\b\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\b")
    assert any(timestamp_pattern.search(block) for block in rendered_blocks)
    expected_timestamp = snapshot.text
    assert any(expected_timestamp in block for block in rendered_blocks)
    assert any(f"🕒 {snapshot.text}" in caption for caption in captions)
    assert time_provider_stub.calls == 2
