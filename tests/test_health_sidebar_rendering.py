"""Tests for the health sidebar rendering without mocking Streamlit."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import textwrap
from typing import Any, Dict

import streamlit as st
from streamlit.runtime.secrets import Secrets
from streamlit.testing.v1 import AppTest

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


def test_sidebar_formats_populated_metrics() -> None:
    base = datetime(2024, 1, 2, 3, 4, 5)
    timestamps = [base.timestamp() + offset for offset in range(6)]

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
        "✅ Refresh correcto • 02/01/2024 03:04:05 — OK",
        "#### 📈 Yahoo Finance",
        "♻️ Fallback local • 02/01/2024 03:04:06 — respaldo",
        "#### 💱 FX",
        "⚠️ API FX con errores • 02/01/2024 03:04:07 (123 ms) — boom",
        "♻️ Uso de caché • 02/01/2024 03:04:08 (edad 46s)",
        "#### ⏱️ Latencias",
        "- Portafolio: 457 ms • fuente: api • fresh • 02/01/2024 03:04:09",
        "- Cotizaciones: 789 ms • fuente: yfinance • items: 12 • with gaps • 02/01/2024 03:04:10",
    }

    missing = expected_lines.difference(markdown)
    assert not missing, f"Missing sidebar lines: {missing}"
