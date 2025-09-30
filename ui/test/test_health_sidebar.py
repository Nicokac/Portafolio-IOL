from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import services.health as health_service
import ui.health_sidebar as health_sidebar
from shared.ui import notes as shared_notes
from shared.version import __version__


def _mock_sidebar(monkeypatch):
    sidebar = SimpleNamespace(
        header=MagicMock(),
        caption=MagicMock(),
        markdown=MagicMock(),
    )
    monkeypatch.setattr(health_sidebar, "st", SimpleNamespace(sidebar=sidebar))
    return sidebar


def _mock_metrics(monkeypatch, metrics):
    monkeypatch.setattr(health_service, "get_health_metrics", lambda: metrics)
    monkeypatch.setattr(health_sidebar, "get_health_metrics", health_service.get_health_metrics)


def _collect_markdown(sidebar):
    return [call.args[0] for call in sidebar.markdown.call_args_list]


def test_render_health_sidebar_with_success_metrics(monkeypatch):
    calls: list[str] = []

    def _spy(note: str) -> str:
        calls.append(note)
        return shared_notes.format_note(note)

    monkeypatch.setattr(health_sidebar, "format_note", _spy)

    metrics = {
        "iol_refresh": {"status": "success", "detail": "Tokens OK", "ts": 1},
        "yfinance": {"source": "fallback", "detail": "AAPL", "ts": 2},
        "fx_api": {
            "status": "success",
            "error": None,
            "elapsed_ms": 123.4,
            "ts": 3,
        },
        "fx_cache": {"mode": "hit", "age": 5.0, "ts": 4},
        "portfolio": {"elapsed_ms": 200.0, "source": "api", "ts": 5},
        "quotes": {
            "elapsed_ms": 350.0,
            "source": "fallback",
            "count": 3,
            "detail": "cache",
            "ts": 6,
        },
    }
    sidebar = _mock_sidebar(monkeypatch)
    _mock_metrics(monkeypatch, metrics)

    health_sidebar.render_health_sidebar()

    sidebar.header.assert_called_once_with(
        f"🩺 Healthcheck (versión {__version__})"
    )

    md_calls = _collect_markdown(sidebar)
    assert any("#### 🔐 Conexión IOL" in text for text in md_calls)
    assert any(":white_check_mark:" in text and "Tokens OK" in text for text in md_calls)
    assert any(":information_source:" in text and "Fallback local" in text for text in md_calls)
    assert any(":white_check_mark:" in text and "API FX OK" in text for text in md_calls)
    assert any(":white_check_mark:" in text and "Uso de caché" in text for text in md_calls)
    assert any("- Portafolio: 200 ms" in text for text in md_calls)
    assert any("- Cotizaciones: 350 ms" in text and "items: 3" in text for text in md_calls)

    assert len(calls) == 4
    assert calls[0].startswith("✅ Refresh correcto •")
    assert calls[0].endswith("Tokens OK")
    assert calls[1].startswith("ℹ️ Fallback local •")
    assert calls[1].endswith("AAPL")
    assert calls[2].startswith("✅ API FX OK •")
    assert "(123 ms)" in calls[2]
    assert calls[3].startswith("✅ Uso de caché •")
    assert calls[3].endswith("(edad 5s)")


def test_render_health_sidebar_with_missing_metrics(monkeypatch):
    calls: list[str] = []

    def _spy(note: str) -> str:
        calls.append(note)
        return shared_notes.format_note(note)

    monkeypatch.setattr(health_sidebar, "format_note", _spy)

    metrics = {
        "iol_refresh": {"status": "error", "detail": "Token inválido", "ts": 10},
        "yfinance": None,
        "fx_api": {
            "status": "error",
            "error": "timeout",
            "elapsed_ms": None,
            "ts": 11,
        },
        "fx_cache": None,
        "portfolio": None,
        "quotes": {
            "elapsed_ms": None,
            "source": "error",
            "detail": "sin datos",
            "ts": None,
        },
    }
    sidebar = _mock_sidebar(monkeypatch)
    _mock_metrics(monkeypatch, metrics)

    health_sidebar.render_health_sidebar()

    sidebar.header.assert_called_once_with(
        f"🩺 Healthcheck (versión {__version__})"
    )

    md_calls = _collect_markdown(sidebar)
    assert any(":warning:" in text and "Error al refrescar" in text for text in md_calls)
    assert "_Sin consultas registradas._" in md_calls
    assert any(":warning:" in text and "API FX con errores" in text for text in md_calls)
    assert "_Sin uso de caché registrado._" in md_calls
    assert any(text.startswith("- Portafolio: sin registro") for text in md_calls)
    assert any("- Cotizaciones: s/d" in text and "sin datos" in text for text in md_calls)

    assert len(calls) == 2
    assert calls[0].startswith("⚠️ Error al refrescar •")
    assert calls[0].endswith("Token inválido")
    assert calls[1].startswith("⚠️ API FX con errores •")
    assert calls[1].endswith("timeout")
