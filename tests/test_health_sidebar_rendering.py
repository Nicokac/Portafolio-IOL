"""Tests for the health sidebar rendering using local Streamlit stubs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import importlib
import sys

import pytest
from zoneinfo import ZoneInfo

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.time_provider import TIME_FORMAT, TimeProvider, TimeSnapshot
from shared.version import __version__
from shared.ui import notes as shared_notes


@pytest.fixture
def health_sidebar_module(monkeypatch: pytest.MonkeyPatch):
    import ui.health_sidebar as health_sidebar

    module = importlib.reload(health_sidebar)
    monkeypatch.setattr(module, "get_health_metrics", lambda: module.st.session_state.get("health_metrics", {}))
    return module


def _run_sidebar(module, metrics: Dict[str, Any]) -> None:
    module.st.session_state["health_metrics"] = metrics
    module.render_health_sidebar()


def test_sidebar_shows_empty_state_labels(streamlit_stub, health_sidebar_module) -> None:
    _run_sidebar(
        health_sidebar_module,
        {
            "iol_refresh": None,
            "yfinance": None,
            "fx_api": None,
            "fx_cache": None,
            "portfolio": None,
            "quotes": None,
            "opportunities": None,
        },
    )

    assert health_sidebar_module.st.sidebar.headers == [
        f"🩺 Healthcheck (versión {__version__})"
    ]
    captions = list(health_sidebar_module.st.sidebar.captions)
    assert any(
        "Monitorea la procedencia" in caption for caption in captions
    ), "Expected introductory caption to be present"

    markdown_calls = list(health_sidebar_module.st.sidebar.markdowns)
    for expected in [
        "#### 🔐 Conexión IOL",
        "_Sin actividad registrada._",
        "#### 📈 Yahoo Finance",
        "_Sin consultas registradas._",
        "#### 💱 FX",
        "_Sin llamadas a la API FX._",
        "_Sin uso de caché registrado._",
        "#### 🔎 Screening de oportunidades",
        "_Sin screenings recientes._",
        "#### ⏱️ Latencias",
        "#### 🧭 Monitoreo de sesiones",
        "_Sin métricas de sesiones._",
        "#### 🧪 Diagnóstico inicial",
        "_Sin diagnósticos registrados._",
        "#### 📄 Logs",
        "_No se encontró analysis.log._",
    ]:
        assert expected in markdown_calls

    assert any(
        "Portafolio: sin registro" in text for text in markdown_calls
    ), "Expected portfolio latency placeholder"
    assert any(
        "Cotizaciones: sin registro" in text for text in markdown_calls
    ), "Expected quotes latency placeholder"


def test_sidebar_formats_populated_metrics(monkeypatch, streamlit_stub, health_sidebar_module) -> None:
    timezone = ZoneInfo("America/Argentina/Buenos_Aires")
    base = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone)
    timestamps = [base.timestamp() + offset for offset in range(7)]

    class StubTimeProvider:
        def __init__(self) -> None:
            self.calls: list[float | None] = []

        def from_timestamp(self, ts):
            if not ts:
                self.calls.append(ts)
                return None
            ts_value = float(ts)
            self.calls.append(ts_value)
            moment = datetime.fromtimestamp(ts_value, tz=timezone)
            return TimeSnapshot(moment.strftime(TIME_FORMAT), moment)

    provider_stub = StubTimeProvider()
    monkeypatch.setattr(health_sidebar_module, "TimeProvider", provider_stub)

    metrics = {
        "iol_refresh": {
            "status": "success",
            "ts": timestamps[0],
            "detail": "OK",
        },
        "yfinance": {
            "source": "fallback",
            "detail": "respaldo",
            "ts": timestamps[1],
        },
        "fx_api": {
            "status": "error",
            "error": "boom",
            "elapsed_ms": 123.4,
            "ts": timestamps[2],
        },
        "fx_cache": {
            "mode": "hit",
            "age": 45.6,
            "ts": timestamps[3],
        },
        "opportunities": {
            "mode": "hit",
            "elapsed_ms": 12.3,
            "cached_elapsed_ms": 45.6,
            "universe_initial": 150,
            "universe_final": 90,
            "discard_ratio": 0.4,
            "highlighted_sectors": ["Energy", "Utilities"],
            "counts_by_origin": {"nyse": 45, "nasdaq": 45},
            "ts": timestamps[4],
        },
        "portfolio": {
            "elapsed_ms": 456.7,
            "source": "api",
            "detail": "fresh",
            "ts": timestamps[5],
        },
        "quotes": {
            "elapsed_ms": 789.1,
            "source": "yfinance",
            "count": 12,
            "detail": "with gaps",
            "ts": timestamps[6],
        },
    }

    _run_sidebar(health_sidebar_module, metrics)

    markdown = list(health_sidebar_module.st.sidebar.markdowns)
    formatted = [str(TimeProvider.from_timestamp(ts)) for ts in timestamps]

    expected_headers = {
        "#### 🔐 Conexión IOL",
        "#### 📈 Yahoo Finance",
        "#### 💱 FX",
        "#### 🔎 Screening de oportunidades",
        shared_notes.format_note(
            "✅ Cache reutilizada • "
            f"{formatted[4]} (12 ms • previo 46 ms) — universo 150→90 | descartes 40% | sectores: Energy, Utilities | origen: nyse=45, nasdaq=45"
        ),
        "#### ⏱️ Latencias",
        "#### 🧭 Monitoreo de sesiones",
        "_Sin métricas de sesiones._",
        "#### 🧪 Diagnóstico inicial",
        "_Sin diagnósticos registrados._",
        "#### 📄 Logs",
        "_No se encontró analysis.log._",
    }

    missing = expected_headers.difference(markdown)
    assert not missing, f"Missing sidebar lines: {missing}"

    assert any(
        "Refresh correcto" in text and formatted[0] in text for text in markdown
    ), "Expected IOL refresh summary"
    assert any(
        "Fallback local" in text and formatted[1] in text for text in markdown
    ), "Expected Yahoo Finance summary"
    assert any(
        "API FX con errores" in text and formatted[2] in text for text in markdown
    ), "Expected FX API summary"
    assert any(
        "Uso de caché" in text and formatted[3] in text for text in markdown
    ), "Expected FX cache summary"
    assert any(
        "Portafolio: 457 ms" in text and formatted[5] in text for text in markdown
    ), "Expected portfolio latency entry"
    assert any(
        "Cotizaciones: 789 ms" in text and formatted[6] in text for text in markdown
    ), "Expected quotes latency entry"
    assert len(provider_stub.calls) == len(timestamps)
    for call, expected in zip(provider_stub.calls, timestamps):
        assert call == pytest.approx(expected)


def test_tab_latency_metrics_include_p99_and_budget(
    streamlit_stub, health_sidebar_module
) -> None:
    metrics = {
        "tab_latencies": {
            "analisis": {
                "label": "Análisis",
                "avg": 180.0,
                "percentiles": {
                    "p50": 150.0,
                    "p90": 200.0,
                    "p95": 210.0,
                    "p99": 240.0,
                },
                "status_counts": {"success": 5},
                "status_ratios": {"success": 1.0},
                "total": 5,
                "error_count": 0,
                "error_ratio": 0.0,
                "error_budget": 0.0,
                "missing_count": 0,
            },
            "riesgos": {
                "label": "Riesgos",
                "avg": 420.0,
                "percentiles": {
                    "p50": 400.0,
                    "p90": 450.0,
                    "p95": 480.0,
                    "p99": 550.0,
                },
                "status_counts": {"success": 6, "error": 4},
                "status_ratios": {"success": 0.6, "error": 0.4},
                "total": 10,
                "error_count": 4,
                "error_ratio": 0.4,
                "error_budget": 0.4,
                "missing_count": 0,
            },
        }
    }

    _run_sidebar(health_sidebar_module, metrics)

    expanders = health_sidebar_module.st.get_records("expander")
    latency_expander = next(
        entry for entry in expanders if entry.get("label") == "Latencias por pestaña"
    )
    latency_lines = [
        child.get("text")
        for child in latency_expander.get("children", [])
        if isinstance(child, dict) and child.get("type") == "markdown"
    ]

    assert any("P99 240 ms" in line for line in latency_lines)
    assert any(":green[Budget 0%]" in line for line in latency_lines)
    assert any(":red[Budget 40%]" in line for line in latency_lines)
