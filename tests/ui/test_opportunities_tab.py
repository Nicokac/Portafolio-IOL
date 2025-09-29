from __future__ import annotations

from pathlib import Path
from types import ModuleType
import sys
import textwrap
from unittest.mock import MagicMock, patch

import pandas as pd
import streamlit as _streamlit_module
from streamlit.runtime.secrets import Secrets
from streamlit.testing.v1 import AppTest
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _resolve_streamlit_module() -> ModuleType:
    """Ensure we use the real Streamlit implementation, not a test stub."""
    if getattr(_streamlit_module, "__file__", None) and hasattr(
        _streamlit_module, "button"
    ):
        return _streamlit_module

    for name in list(sys.modules):
        if name == "streamlit" or name.startswith("streamlit."):
            sys.modules.pop(name, None)

    import streamlit as real_streamlit

    return real_streamlit


def _normalize_streamlit_module() -> None:
    global st
    if sys.modules.get("streamlit") is not _ORIGINAL_STREAMLIT:
        sys.modules["streamlit"] = _ORIGINAL_STREAMLIT
    if st is not _ORIGINAL_STREAMLIT:
        st = _ORIGINAL_STREAMLIT
    ui_settings_mod = sys.modules.get("ui.ui_settings")
    if ui_settings_mod is not None:
        setattr(ui_settings_mod, "st", _ORIGINAL_STREAMLIT)


def _render_app() -> AppTest:
    _normalize_streamlit_module()
    if not hasattr(st, "secrets"):
        st.secrets = Secrets({})
    app = AppTest.from_string(_SCRIPT)
    app.run()
    return app


def _run_app_with_result(result: dict[str, object]) -> tuple[AppTest, MagicMock]:
    _normalize_streamlit_module()
    if not hasattr(st, "secrets"):
        st.secrets = Secrets({})
    app = AppTest.from_string(_SCRIPT)
    patch_target = "controllers.opportunities.generate_opportunities_report"
    with patch(patch_target, return_value=result) as mock:
        app.run()
        buttons = [
            button for button in app.get("button") if getattr(button, "key", None) == "search_opportunities"
        ]
        assert (
            buttons
        ), "Expected the search button with key 'search_opportunities' to be rendered"
        buttons[0].click()
        app.run()
    return app, mock


st = _resolve_streamlit_module()
sys.modules["streamlit"] = st
_ORIGINAL_STREAMLIT = st


@pytest.fixture(autouse=True)
def _ensure_opportunities_flag_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the opportunities tab feature flag enabled by default in this suite."""

    import shared.config as shared_config

    monkeypatch.setattr("shared.settings.FEATURE_OPPORTUNITIES_TAB", True)
    monkeypatch.setenv("FEATURE_OPPORTUNITIES_TAB", "true")
    monkeypatch.setattr(shared_config.settings, "tokens_key", "dummy", raising=False)
    monkeypatch.setattr(shared_config.settings, "allow_plain_tokens", True, raising=False)

from shared.version import __version__

_SCRIPT = textwrap.dedent(
    f"""
    import sys
    sys.path.insert(0, {repr(str(_PROJECT_ROOT))})
    from ui.tabs.opportunities import render_opportunities_tab
    render_opportunities_tab()
    """
)


def test_header_displays_version() -> None:
    app = _render_app()
    headers = [element.value for element in app.get("header")]
    expected_header = f"🚀 Empresas con oportunidad · beta {__version__}"
    assert expected_header in headers


def test_button_executes_controller_and_shows_yahoo_note() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "price": [180.12, 325.74],
            "score_compuesto": [8.5, 7.9],
        }
    )
    app, mock = _run_app_with_result({"table": df, "notes": []})
    assert mock.call_count == 1
    called_with = mock.call_args.args[0]
    assert called_with == {
        "min_market_cap": 500.0,
        "max_pe": 25.0,
        "min_revenue_growth": 5.0,
        "include_latam": True,
    }
    dataframes = app.get("arrow_data_frame")
    assert dataframes, "Expected Streamlit dataframe component after execution"
    captions = [element.value for element in app.get("caption")]
    assert (
        "Resultados obtenidos de Yahoo Finance (con fallback a datos simulados si falta información)."
        in captions
    )
    assert (
        "ℹ️ Los filtros avanzados de capitalización, P/E, crecimiento e inclusión de Latam requieren datos en vivo de Yahoo."
        in captions
    )


def test_fallback_note_is_displayed_when_present() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["KO"],
            "price": [58.31],
            "score_compuesto": [6.1],
        }
    )
    fallback_note = "⚠️ Datos simulados (Yahoo no disponible)"
    app, _ = _run_app_with_result({"table": df, "notes": [fallback_note]})
    markdown_blocks = [element.value for element in app.get("markdown")]
    assert any(fallback_note in block for block in markdown_blocks)


def test_opportunities_tab_not_rendered_when_flag_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shared.settings.FEATURE_OPPORTUNITIES_TAB", False)
    monkeypatch.delenv("FEATURE_OPPORTUNITIES_TAB", raising=False)

    import controllers.portfolio as portfolio_controller
    import controllers.auth as auth_controller
    import services.cache as cache_services
    import ui.actions as actions_module
    import ui.footer as footer_module
    import ui.header as header_module
    import ui.health_sidebar as health_module
    import ui.login as login_module

    monkeypatch.setattr(portfolio_controller, "render_portfolio_section", lambda *a, **k: None)
    monkeypatch.setattr(auth_controller, "build_iol_client", lambda: object())
    monkeypatch.setattr(cache_services, "get_fx_rates_cached", lambda: ({}, None))
    monkeypatch.setattr(header_module, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(actions_module, "render_action_menu", lambda *a, **k: None)
    monkeypatch.setattr(footer_module, "render_footer", lambda *a, **k: None)
    monkeypatch.setattr(health_module, "render_health_sidebar", lambda *a, **k: None)
    monkeypatch.setattr(login_module, "render_login_page", lambda *a, **k: None)

    app = AppTest.from_file("app.py")
    app.session_state["authenticated"] = True
    app.run()

    assert not app.get("tabs"), "Expected opportunities tabs to be absent when flag is disabled"
    headers = [element.value for element in app.get("header")]
    assert all("Empresas con oportunidad" not in header for header in headers)
