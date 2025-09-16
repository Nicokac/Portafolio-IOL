"""Tests for the health sidebar rendering without mocking Streamlit."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys
import textwrap
from typing import Any, Dict

import pytest
from zoneinfo import ZoneInfo

import streamlit as st
from streamlit.runtime.secrets import Secrets
from streamlit.testing.v1 import AppTest

from shared.time_provider import TimeSnapshot
from shared.version import __version__

_ORIGINAL_STREAMLIT = st
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = textwrap.dedent(
    f"""
    import sys
    sys.path.insert(0, {repr(str(_PROJECT_ROOT))})
    from ui.health_sidebar import render_health_sidebar
    render_health_sidebar()
    """
)


def _ensure_real_streamlit_module() -> None:
    """Restore the actual Streamlit module if previous tests replaced it."""
    global st
    if sys.modules.get("streamlit") is not _ORIGINAL_STREAMLIT:
        sys.modules["streamlit"] = _ORIGINAL_STREAMLIT
    if st is not _ORIGINAL_STREAMLIT:
        st = _ORIGINAL_STREAMLIT  # pragma: no cover - simple assignment for tests


def _run_sidebar(metrics: Dict[str, Any]) -> AppTest:
    """Execute the sidebar renderer with the provided metrics."""
    _ensure_real_streamlit_module()
    if not hasattr(st, "secrets"):
        st.secrets = Secrets({})
    app = AppTest.from_string(_SCRIPT)
    app.session_state["health_metrics"] = metrics
    app.run()
    return app


def _collect(app: AppTest, element_type: str) -> list[str]:
    return [element.value for element in app.sidebar if element.type == element_type]


def test_sidebar_shows_empty_state_labels() -> None:
    app = _run_sidebar(
        {
            "iol_refresh": None,
            "yfinance": None,
            "fx_api": None,
            "fx_cache": None,
            "portfolio": None,
            "quotes": None,
        }
    )

    assert _collect(app, "header") == [
        f"🩺 Healthcheck (versión {__version__})"
    ]
    assert "Monitorea la procedencia y el rendimiento de los datos cargados." in _collect(
        app, "caption"
    )

    markdown = _collect(app, "markdown")
    for expected in [
        "#### 🔐 Conexión IOL",
        "_Sin actividad registrada._",
        "#### 📈 Yahoo Finance",
        "_Sin consultas registradas._",
        "#### 💱 FX",
        "_Sin llamadas a la API FX._",
        "_Sin uso de caché registrado._",
        "#### ⏱️ Latencias",
        "- Portafolio: sin registro",
        "- Cotizaciones: sin registro",
    ]:
        assert expected in markdown


def test_sidebar_formats_populated_metrics(monkeypatch) -> None:
    timezone = ZoneInfo("America/Argentina/Buenos_Aires")
    base = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone)
    timestamps = [base.timestamp() + offset for offset in range(6)]

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
            return TimeSnapshot(moment.strftime("%Y-%m-%d %H:%M:%S"), moment)

    provider_stub = StubTimeProvider()
    monkeypatch.setattr("ui.health_sidebar.TimeProvider", provider_stub)

    def fmt(offset: int) -> str:
        return (base + timedelta(seconds=offset)).strftime("%Y-%m-%d %H:%M:%S")

    app = _run_sidebar(
        {
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
            "portfolio": {
                "elapsed_ms": 456.7,
                "source": "api",
                "detail": "fresh",
                "ts": timestamps[4],
            },
            "quotes": {
                "elapsed_ms": 789.1,
                "source": "yfinance",
                "count": 12,
                "detail": "with gaps",
                "ts": timestamps[5],
            },
        }
    )

    markdown = _collect(app, "markdown")
    expected_lines = {
        "#### 🔐 Conexión IOL",
        f"✅ Refresh correcto • {fmt(0)} — OK",
        "#### 📈 Yahoo Finance",
        f"♻️ Fallback local • {fmt(1)} — respaldo",
        "#### 💱 FX",
        f"⚠️ API FX con errores • {fmt(2)} (123 ms) — boom",
        f"♻️ Uso de caché • {fmt(3)} (edad 46s)",
        "#### ⏱️ Latencias",
        f"- Portafolio: 457 ms • fuente: api • fresh • {fmt(4)}",
        f"- Cotizaciones: 789 ms • fuente: yfinance • items: 12 • with gaps • {fmt(5)}",
    }

    missing = expected_lines.difference(markdown)
    assert not missing, f"Missing sidebar lines: {missing}"
    assert len(provider_stub.calls) == len(timestamps)
    for call, expected in zip(provider_stub.calls, timestamps):
        assert call == pytest.approx(expected)
