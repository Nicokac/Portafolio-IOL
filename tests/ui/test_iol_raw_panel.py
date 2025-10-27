from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from tests.fixtures.streamlit import UIFakeStreamlit


@pytest.fixture(autouse=True)
def _silence_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("application.portfolio_service.log_metric", lambda *_, **__: None)


def _load_panel(monkeypatch: pytest.MonkeyPatch, streamlit_stub: UIFakeStreamlit):
    module = importlib.import_module("ui.panels.iol_raw_debug")
    module = importlib.reload(module)
    monkeypatch.setattr(module, "st", streamlit_stub)
    return module


def test_render_panel_captures_snapshot_and_renders_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    streamlit_stub = UIFakeStreamlit(button_clicks={"🔍 Capturar IOL RAW (BPOC7)": [True]})
    json_calls: list[Any] = []

    def fake_json(data, **_):
        json_calls.append(data)

    streamlit_stub.json = fake_json  # type: ignore[attr-defined]
    streamlit_stub.session_state["cli"] = object()
    streamlit_stub.session_state["portfolio_last_viewmodel"] = SimpleNamespace(
        positions=pd.DataFrame({"simbolo": ["BPOC7"], "cantidad": [146], "ultimo": [1.0]})
    )

    snapshot = {
        "ts": "2025-01-01T00:00:00+00:00",
        "portfolio_raw": {"activos": [{"simbolo": "BPOC7", "cantidad": 146}]},
        "portfolio_row": {"simbolo": "BPOC7", "cantidad": 146},
        "quote_raw": {"ultimoPrecio": 1377.0},
        "quote_detail_raw": {"ultimoPrecio": 1377.0, "detalle": True},
    }

    module = _load_panel(monkeypatch, streamlit_stub)
    monkeypatch.setattr(module, "capture_iol_raw_snapshot", lambda *_, **__: snapshot)
    monkeypatch.setattr(
        module,
        "to_iol_format",
        lambda df: pd.DataFrame({"Activo": ["BPOC7"], "Cantidad": [146]}),
    )

    module.render_iol_raw_debug_panel()

    assert streamlit_stub.session_state["iol_raw_last_snapshot"] == snapshot
    assert len(streamlit_stub.spinner_messages) == 1
    assert json_calls[:4]
    assert streamlit_stub.download_buttons
    file_name = streamlit_stub.download_buttons[0]["file_name"]
    assert file_name == "iol_raw_BPOC7_2025-01-01T000000+0000.json"
    assert streamlit_stub.dataframes


def test_render_panel_shows_warning_when_symbol_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    streamlit_stub = UIFakeStreamlit()
    json_calls: list[Any] = []

    def fake_json(data, **_):
        json_calls.append(data)

    streamlit_stub.json = fake_json  # type: ignore[attr-defined]
    snapshot = {
        "ts": "2025-01-01T00:00:00+00:00",
        "portfolio_raw": {"activos": [{"simbolo": "GGAL", "cantidad": 5}]},
        "portfolio_row": None,
        "quote_raw": {"ultimoPrecio": 200.0},
        "quote_detail_raw": {"ultimoPrecio": 200.0},
    }
    streamlit_stub.session_state["iol_raw_last_snapshot"] = snapshot

    module = _load_panel(monkeypatch, streamlit_stub)
    monkeypatch.setattr(
        module,
        "to_iol_format",
        lambda df: pd.DataFrame({"Activo": ["GGAL"], "Cantidad": [5]}),
    )

    module.render_iol_raw_debug_panel()

    assert streamlit_stub.warnings
    assert json_calls  # quote payloads still rendered
    assert streamlit_stub.download_buttons
