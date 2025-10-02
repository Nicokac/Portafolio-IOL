from __future__ import annotations

from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace
from typing import Iterable

import streamlit as _streamlit_module
from streamlit.runtime.secrets import Secrets
from streamlit.testing.v1 import AppTest
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import services.health as health_service  # noqa: E402  - added to sys.path above
import ui.health_sidebar as health_sidebar  # noqa: E402  - added to sys.path above


def _resolve_streamlit_module():
    if getattr(_streamlit_module, "__file__", None) and hasattr(_streamlit_module, "sidebar"):
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
    if getattr(health_sidebar, "st", None) is not _ORIGINAL_STREAMLIT:
        health_sidebar.st = _ORIGINAL_STREAMLIT


st = _resolve_streamlit_module()
sys.modules["streamlit"] = st
_ORIGINAL_STREAMLIT = st


class _DummySidebar:
    def __init__(self) -> None:
        self.headers: list[str] = []
        self.captions: list[str] = []
        self.markdown_calls: list[str] = []

    def header(self, text: str) -> None:
        self.headers.append(text)

    def caption(self, text: str) -> None:
        self.captions.append(text)

    def markdown(self, text: str) -> None:
        self.markdown_calls.append(text)


class _DummyStreamlit:
    def __init__(self) -> None:
        self.sidebar = _DummySidebar()


@pytest.fixture
def _dummy_metrics() -> dict[str, object]:
    return {
        "iol_refresh": {"status": "success", "detail": "OK", "ts": None},
        "yfinance": {"source": "yfinance", "detail": "cache", "ts": None},
        "fx_api": {"status": "error", "error": "boom", "elapsed_ms": 250.5, "ts": None},
        "fx_cache": {"mode": "hit", "age": 12.3, "ts": None},
        "opportunities": {
            "mode": "miss",
            "elapsed_ms": 321.0,
            "cached_elapsed_ms": None,
            "universe_initial": 120,
            "universe_final": 48,
            "discard_ratio": 0.6,
            "highlighted_sectors": ["Energy", "Utilities"],
            "counts_by_origin": {"nyse": 30, "nasdaq": 18},
            "ts": None,
        },
        "portfolio": {
            "elapsed_ms": 123.4,
            "source": "api",
            "detail": "fresh",
            "ts": None,
        },
        "quotes": {
            "elapsed_ms": 456.7,
            "source": "yfinance",
            "count": 5,
            "detail": "gap",
            "ts": None,
        },
        "opportunities_history": [
            {
                "mode": "miss",
                "elapsed_ms": 400.0,
                "cached_elapsed_ms": 250.0,
                "ts": None,
            },
            {
                "mode": "hit",
                "elapsed_ms": 220.0,
                "cached_elapsed_ms": 210.0,
                "ts": None,
            },
        ],
    }


def test_render_health_sidebar_uses_shared_note_formatter(
    monkeypatch: pytest.MonkeyPatch, _dummy_metrics: dict[str, object]
) -> None:
    dummy_streamlit = _DummyStreamlit()
    monkeypatch.setattr(health_sidebar, "st", dummy_streamlit)
    monkeypatch.setattr(health_sidebar, "get_health_metrics", lambda: _dummy_metrics)

    captured: list[str] = []

    def _fake_format(note: str) -> str:
        captured.append(note)
        return f"formatted::{note}"

    monkeypatch.setattr(health_sidebar, "format_note", _fake_format)
    monkeypatch.setattr(health_sidebar.shared_notes, "format_note", _fake_format)

    health_sidebar.render_health_sidebar()

    render_call_count = len(captured)

    fx_lines = list(
        health_sidebar._format_fx_section(
            _dummy_metrics["fx_api"], _dummy_metrics["fx_cache"]
        )
    )
    latency_lines = list(
        health_sidebar._format_latency_section(
            _dummy_metrics["portfolio"], _dummy_metrics["quotes"]
        )
    )
    opportunities_note = health_sidebar._format_opportunities_status(
        _dummy_metrics["opportunities"], _dummy_metrics["opportunities_history"]
    )
    history_lines = list(
        health_sidebar._format_opportunities_history(
            reversed(_dummy_metrics["opportunities_history"])
        )
    )
    assert "universo 120→48" in opportunities_note
    assert "descartes 60%" in opportunities_note
    assert "sectores: Energy, Utilities" in opportunities_note
    assert "origen: nyse=30, nasdaq=18" in opportunities_note
    formatted_latency_lines = [f"formatted::{line}" for line in latency_lines]
    expected_notes: list[str] = [
        health_sidebar._format_iol_status(_dummy_metrics["iol_refresh"]),
        health_sidebar._format_yfinance_status(_dummy_metrics["yfinance"]),
        *fx_lines,
        opportunities_note,
        *history_lines,
        *formatted_latency_lines,
    ]

    if len(captured) > render_call_count:
        del captured[render_call_count:]

    expected_raw_notes = [note.removeprefix("formatted::") for note in expected_notes]

    assert captured == expected_raw_notes

    expected_markdown_sequence: list[str] = [
        "#### 🔐 Conexión IOL",
        expected_notes[0],
        "#### 📈 Yahoo Finance",
        expected_notes[1],
        "#### 💱 FX",
        *fx_lines,
        "#### 🔎 Screening de oportunidades",
        opportunities_note,
        "#### 🗂️ Historial de screenings",
        *history_lines,
        "#### ⏱️ Latencias",
        *formatted_latency_lines,
    ]

    assert dummy_streamlit.sidebar.markdown_calls == expected_markdown_sequence


def test_format_opportunities_status_handles_partial_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(health_sidebar, "format_note", lambda text: text)
    data = {
        "mode": "miss",
        "elapsed_ms": None,
        "cached_elapsed_ms": None,
        "universe_final": "15",
        "discard_ratio": "not-a-number",
        "highlighted_sectors": "Energy",
        "counts_by_origin": {"nyse": "10", "": 5, "invalid": "oops"},
        "ts": None,
    }

    note = health_sidebar._format_opportunities_status(data)

    assert "universo final 15" in note
    assert "descartes" not in note
    assert "sectores: Energy" in note
    assert "origen: nyse=10" in note
    assert "s/d" in note


def test_render_health_sidebar_includes_history_section(
    monkeypatch: pytest.MonkeyPatch, _dummy_metrics: dict[str, object]
) -> None:
    dummy_streamlit = _DummyStreamlit()
    monkeypatch.setattr(health_sidebar, "st", dummy_streamlit)
    monkeypatch.setattr(health_sidebar, "get_health_metrics", lambda: _dummy_metrics)

    health_sidebar.render_health_sidebar()

    assert "#### 🗂️ Historial de screenings" in dummy_streamlit.sidebar.markdown_calls
    expected_history_lines = list(
        health_sidebar._format_opportunities_history(
            reversed(_dummy_metrics["opportunities_history"])
        )
    )
    for line in expected_history_lines:
        assert line in dummy_streamlit.sidebar.markdown_calls


def test_record_opportunities_report_rotates_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_state: dict[str, object] = {}
    monkeypatch.setattr(
        health_service,
        "st",
        SimpleNamespace(session_state=dummy_state),
    )

    for idx in range(health_service._OPPORTUNITIES_HISTORY_LIMIT + 2):
        health_service.record_opportunities_report(
            mode="hit" if idx % 2 == 0 else "miss",
            elapsed_ms=100 + idx,
            cached_elapsed_ms=50 + idx,
        )

    metrics = health_service.get_health_metrics()
    history = metrics["opportunities_history"]
    assert len(history) == health_service._OPPORTUNITIES_HISTORY_LIMIT
    last_entry = metrics["opportunities"]
    assert history[-1] == last_entry


_SMOKE_SCRIPT = textwrap.dedent(
    f"""
    import sys
    sys.path.insert(0, {repr(str(_PROJECT_ROOT))})
    import streamlit as st
    from ui.health_sidebar import render_health_sidebar

    tabs = st.tabs(["Resumen", "Detalle", "Histórico"])
    for index, tab in enumerate(tabs):
        with tab:
            st.write(f"Tab {{index}}")

    render_health_sidebar()
    """
)


def test_health_sidebar_smoke_renders_across_tabs(
    monkeypatch: pytest.MonkeyPatch, _dummy_metrics: dict[str, dict[str, object]]
) -> None:
    _normalize_streamlit_module()
    if not hasattr(st, "secrets"):
        st.secrets = Secrets({})

    monkeypatch.setattr("services.health.get_health_metrics", lambda: _dummy_metrics)
    monkeypatch.setattr(health_sidebar, "get_health_metrics", lambda: _dummy_metrics)

    app = AppTest.from_string(_SMOKE_SCRIPT)
    app.run()

    sidebar_markdown = [element.value for element in app.sidebar if element.type == "markdown"]

    def _collect_main_markdown() -> Iterable[str]:
        return [element.value for element in app.get("markdown") if element.value.startswith("Tab ")]

    assert any(value.startswith(":white_check_mark:") for value in sidebar_markdown)
    assert any(value.startswith(":warning:") for value in sidebar_markdown)
    assert len(list(_collect_main_markdown())) == 3
