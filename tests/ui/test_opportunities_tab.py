from __future__ import annotations

from pathlib import Path
from types import ModuleType
import sys
import textwrap
from typing import Callable, Mapping
from unittest.mock import MagicMock, patch

import pandas as pd
import streamlit as _streamlit_module
from streamlit.runtime.secrets import Secrets
from streamlit.testing.v1 import AppTest
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.errors import AppError
import shared.settings as shared_settings

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


def _run_app_with_result(
    result: Mapping[str, object]
    | Callable[[Mapping[str, object]], Mapping[str, object]],
    overrides: Mapping[str, object] | None = None,
) -> tuple[AppTest, MagicMock]:
    _normalize_streamlit_module()
    if not hasattr(st, "secrets"):
        st.secrets = Secrets({})
    app = AppTest.from_string(_SCRIPT)
    patch_target = "controllers.opportunities.generate_opportunities_report"
    with patch(patch_target) as mock:
        if callable(result):
            mock.side_effect = result
        else:
            mock.return_value = result
        app.run()
        if overrides:
            elements = (
                list(app.get("number_input"))
                + list(app.get("slider"))
                + list(app.get("checkbox"))
                + list(app.get("multiselect"))
            )
            for element in elements:
                label = getattr(element, "label", None)
                if label and label in overrides:
                    element.set_value(overrides[label])
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


def test_button_executes_controller_and_shows_yahoo_caption() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT"],
            "price": [180.12, 325.74],
            "score_compuesto": [85.0, 79.0],
            "sector": ["Technology", "Technology"],
        }
    )
    assert "sector" in df.columns
    overrides = {
        "Capitalización mínima (US$ MM)": 750,
        "P/E máximo": 18.5,
        "Crecimiento ingresos mínimo (%)": 12.0,
        "Payout máximo (%)": 65.0,
        "Racha mínima de dividendos (años)": 7,
        "CAGR mínimo de dividendos (%)": 6.5,
        "Crecimiento mínimo de EPS (%)": 4.0,
        "Buyback mínimo (%)": 1.5,
        "Incluir Latam": False,
        "Score mínimo": 72,
        "Sectores": ["Technology"],
    }
    app, mock = _run_app_with_result({"table": df, "notes": [], "source": "yahoo"}, overrides)
    assert mock.call_count == 1
    called_with = mock.call_args.args[0]
    assert called_with == {
        "min_market_cap": 750.0,
        "max_pe": 18.5,
        "min_revenue_growth": 12.0,
        "max_payout": 65.0,
        "min_div_streak": 7,
        "min_cagr": 6.5,
        "min_eps_growth": 4.0,
        "min_buyback": 1.5,
        "include_latam": False,
        "include_technicals": False,
        "min_score_threshold": 72.0,
        "max_results": shared_settings.max_results,
        "sectors": ["Technology"],
    }
    max_results_inputs = [
        element for element in app.get("number_input") if element.label == "Máximo de resultados"
    ]
    assert max_results_inputs, "Expected to find number input for maximum results"
    assert int(max_results_inputs[0].value) == shared_settings.max_results
    dataframes = app.get("arrow_data_frame")
    assert dataframes, "Expected Streamlit dataframe component after execution"
    captions = [element.value for element in app.get("caption")]
    assert any("Yahoo Finance" in caption for caption in captions)
    assert (
        "ℹ️ Los filtros avanzados de capitalización, P/E, crecimiento de ingresos, payout, racha de dividendos, CAGR, crecimiento de EPS, buybacks e inclusión de Latam requieren datos en vivo de Yahoo."
        in captions
    )
    fallback_note = "⚠️ Datos simulados (Yahoo no disponible)"
    markdown_blocks = [element.value for element in app.get("markdown")]
    assert not any(fallback_note in block for block in markdown_blocks)


def test_checkbox_include_technicals_updates_params() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "price": [180.12],
            "score_compuesto": [85.0],
        }
    )
    overrides = {"Incluir indicadores técnicos": True}
    app, mock = _run_app_with_result({"table": df, "notes": [], "source": "yahoo"}, overrides)

    assert mock.call_count == 1
    called_with = mock.call_args.args[0]
    assert called_with["include_technicals"] is True
    assert called_with["include_latam"] is True

    dataframes = app.get("arrow_data_frame")
    assert dataframes, "Expected Streamlit dataframe component after execution"


def test_min_score_slider_uses_settings_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shared_settings, "min_score_threshold", 67)

    app = _render_app()

    sliders = [element for element in app.get("slider") if element.label == "Score mínimo"]
    assert sliders, "Expected to find slider for minimum score"
    assert int(sliders[0].value) == int(shared_settings.min_score_threshold)


def test_min_score_slider_normalizes_out_of_range_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shared_settings, "min_score_threshold", 150)

    app = _render_app()

    sliders = [element for element in app.get("slider") if element.label == "Score mínimo"]
    assert sliders, "Expected to find slider for minimum score"
    assert int(sliders[0].value) == 100


def test_excluded_tickers_not_displayed_even_when_relaxing_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from controllers import opportunities as controller_mod

    excluded = ["MSFT"]
    real_generate = controller_mod.generate_opportunities_report

    monkeypatch.setattr(
        controller_mod,
        "run_screener_yahoo",
        lambda **kwargs: (_ for _ in ()).throw(AppError("Yahoo no disponible")),
    )

    captured_params: list[Mapping[str, object]] = []

    def fake_generate(params: Mapping[str, object]) -> Mapping[str, object]:
        updated = dict(params)
        updated["exclude_tickers"] = excluded
        captured_params.append(updated)
        return real_generate(updated)

    overrides = {
        "Capitalización mínima (US$ MM)": 0,
        "P/E máximo": 60.0,
        "Score mínimo": 50,
    }

    app, mock = _run_app_with_result(fake_generate, overrides)

    assert mock.call_count >= 1
    assert captured_params, "Expected generate_opportunities_report to be invoked"
    for payload in captured_params:
        assert payload.get("exclude_tickers") == excluded

    dataframes = app.get("arrow_data_frame")
    assert dataframes, "Expected Streamlit dataframe component after execution"
    displayed = dataframes[0].value
    assert not displayed.empty
    assert "MSFT" not in set(displayed["ticker"])
    assert "AAPL" in set(displayed["ticker"])


def test_fallback_legend_and_notes_displayed_when_stub_source() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["KO"],
            "price": [58.31],
            "score_compuesto": [61.0],
        }
    )
    fallback_note = "⚠️ Datos simulados (Yahoo no disponible)"
    extra_note = "ℹ️ Recuerda validar con fuentes oficiales"
    app, _ = _run_app_with_result(
        {"table": df, "notes": [fallback_note, extra_note], "source": "stub"}
    )
    captions = [element.value for element in app.get("caption")]
    assert any("Resultados simulados" in caption for caption in captions)
    markdown_blocks = [element.value for element in app.get("markdown")]
    assert any(f"- **{fallback_note}**" in block for block in markdown_blocks)
    assert any(extra_note in block for block in markdown_blocks)


def test_stub_source_displays_warning_caption_and_notes() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["PFE"],
            "price": [35.12],
            "score_compuesto": [54.0],
        }
    )
    extra_note = "Dato adicional relevante"
    app, _ = _run_app_with_result(
        {"table": df, "notes": [extra_note], "source": "stub"}
    )
    captions = [element.value for element in app.get("caption")]
    assert "⚠️ Resultados simulados (Yahoo no disponible)" in captions
    assert not any(
        "Resultados obtenidos de Yahoo Finance" in caption for caption in captions
    )
    assert any(
        "ℹ️ Los filtros avanzados" in caption for caption in captions
    ), "Expected informational caption to remain visible"
    markdown_blocks = [element.value for element in app.get("markdown")]
    assert any(extra_note in block for block in markdown_blocks)


def test_fallback_note_with_cause_highlighted() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["KO"],
            "price": [58.31],
            "score_compuesto": [61.0],
        }
    )
    fallback_note = "⚠️ Datos simulados — Causa: Yahoo timeout"

    app, _ = _run_app_with_result(
        {"table": df, "notes": [fallback_note], "source": "stub"}
    )

    markdown_blocks = [element.value for element in app.get("markdown")]

    assert any(f"- **{fallback_note}**" in block for block in markdown_blocks)


def test_notes_block_highlights_backend_messages() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["AMZN"],
            "price": [140.25],
            "score_compuesto": [78.0],
        }
    )
    top_note = (
        f"Se recortaron los resultados a los {shared_settings.max_results} mejores según el score compuesto."
    )
    threshold_note = "No se encontraron candidatos con score >= 75 tras aplicar el threshold."
    regular_note = "Considerar diversificación adicional."

    app, _ = _run_app_with_result(
        {
            "table": df,
            "notes": [top_note, threshold_note, regular_note],
            "source": "yahoo",
        }
    )

    markdown_blocks = [element.value for element in app.get("markdown")]

    assert f"- **{top_note}**" in markdown_blocks
    assert f"- **{threshold_note}**" in markdown_blocks
    assert f"- {regular_note}" in markdown_blocks


def test_notes_block_highlights_scarcity_messages() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["NFLX"],
            "price": [410.55],
            "score_compuesto": [71.0],
        }
    )
    scarcity_note = "Solo se encontraron 3 candidatos por debajo del mínimo esperado."

    app, _ = _run_app_with_result(
        {"table": df, "notes": [scarcity_note], "source": "yahoo"}
    )

    markdown_blocks = [element.value for element in app.get("markdown")]

    assert f"- **{scarcity_note}**" in markdown_blocks


def test_notes_block_formats_truncation_and_shortage_notes() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["META", "GOOGL"],
            "price": [295.12, 138.45],
            "score_compuesto": [83.5, 79.2],
        }
    )
    truncation_note = "Se muestran 10 resultados de 240 tras aplicar el máximo solicitado (10)."
    shortage_note = "Solo se encontraron 10 oportunidades (mínimo esperado: 25)."

    app, _ = _run_app_with_result(
        {"table": df, "notes": [truncation_note, shortage_note], "source": "yahoo"}
    )

    captions = [element.value for element in app.get("caption")]
    assert any("Yahoo Finance" in caption for caption in captions)

    markdown_blocks = [element.value for element in app.get("markdown")]
    assert f"- **{truncation_note}**" in markdown_blocks
    assert f"- **{shortage_note}**" in markdown_blocks


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
