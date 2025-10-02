from __future__ import annotations

import importlib
from pathlib import Path
from typing import Mapping

import pandas as pd
import pytest
import sys
from types import SimpleNamespace

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.version import __version__


@pytest.fixture
def opportunities_tab(streamlit_stub, monkeypatch: pytest.MonkeyPatch):
    import ui.tabs.opportunities as opportunities_module

    module = importlib.reload(opportunities_module)
    streamlit_stub.session_state.clear()
    streamlit_stub.secrets.clear()
    return module


def _render(module) -> None:
    module.render_opportunities_tab()


def test_header_displays_version(opportunities_tab, streamlit_stub) -> None:
    _render(opportunities_tab)

    headers = [entry["text"] for entry in streamlit_stub.get_records("header")]
    expected_header = f"🚀 Empresas con oportunidad · v{__version__}"
    assert expected_header in headers

    expanders = streamlit_stub.get_records("expander")
    assert any(entry["label"].startswith("¿Qué significa") for entry in expanders)


def test_button_executes_controller_and_updates_summary(
    opportunities_tab, streamlit_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_table = pd.DataFrame(
        {
            "ticker": ["AAPL", "NEE"],
            "Yahoo Finance Link": [
                "https://finance.yahoo.com/quote/AAPL",
                "https://finance.yahoo.com/quote/NEE",
            ],
            "score_compuesto": [85.0, 72.0],
        }
    )
    result_summary = {
        "universe_count": 150,
        "result_count": 2,
        "discarded_ratio": 0.5,
        "selected_sectors": ["Technology", "Utilities"],
        "sector_distribution": {},
    }
    result_notes = ["✅ Screening completado", "⚠️ Revisar filtros"]

    expected_params: Mapping[str, object] | None = None

    def fake_generate(params: Mapping[str, object]):
        nonlocal expected_params
        expected_params = params
        return {
            "table": result_table,
            "summary": result_summary,
            "notes": result_notes,
            "source": "yahoo",
        }

    monkeypatch.setitem(sys.modules, "controllers.opportunities", SimpleNamespace(generate_opportunities_report=fake_generate))
    streamlit_stub.set_button_result("search_opportunities", True)

    _render(opportunities_tab)

    assert expected_params is not None
    assert expected_params["min_market_cap"] == 500
    assert expected_params["max_pe"] == 25.0
    assert streamlit_stub.session_state[opportunities_tab._SUMMARY_STATE_KEY]["result_count"] == 2

    dataframes = streamlit_stub.get_records("dataframe")
    assert dataframes, "Expected results dataframe to be rendered"
    captions = streamlit_stub.get_records("caption")
    assert any("Resultados" in entry["text"] for entry in captions)


def test_compare_presets_invokes_controller_for_each_column(
    opportunities_tab, streamlit_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Mapping[str, object]] = []

    def fake_generate(params: Mapping[str, object]):
        calls.append(params)
        return {
            "table": pd.DataFrame({"ticker": ["AAPL"], "Yahoo Finance Link": ["https://finance.yahoo.com/quote/AAPL"]}),
            "summary": None,
            "notes": None,
            "source": "yahoo",
        }

    monkeypatch.setitem(sys.modules, "controllers.opportunities", SimpleNamespace(generate_opportunities_report=fake_generate))
    streamlit_stub.set_form_submit_result("compare_presets_form", True)
    streamlit_stub.set_form_submit_result("Comparar presets", True)

    _render(opportunities_tab)

    assert calls, "Expected comparison to trigger controller calls"
    assert len(calls) == 2


def test_normalization_helpers_handle_edge_cases(opportunities_tab) -> None:
    assert opportunities_tab._normalize_notes(None) == []
    assert opportunities_tab._normalize_notes("msg") == ["msg"]
    assert opportunities_tab._normalize_notes({"a": "note"}) == ["note"]

    table = pd.DataFrame({"ticker": ["AAPL"]})
    extracted_table, notes, source = opportunities_tab._extract_result({
        "table": table,
        "notes": ["note"],
        "source": "stub",
    })
    assert extracted_table.equals(table)
    assert notes == ["note"]
    assert source == "stub"

    summary = opportunities_tab._normalize_summary_payload({
        "universe_count": 10,
        "discarded_ratio": 0.5,
    })
    assert summary == {"universe_count": 10, "discarded_ratio": 0.5}
