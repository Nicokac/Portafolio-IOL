import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import services.health as health_service
import ui.health_sidebar as health_sidebar
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
    assert any("✅ Refresh correcto" in text and "Tokens OK" in text for text in md_calls)
    assert any("♻️ Fallback local" in text and "AAPL" in text for text in md_calls)
    assert any("API FX OK" in text and "123" in text for text in md_calls)
    assert any("Uso de caché" in text and "edad" in text for text in md_calls)
    assert any("- Portafolio: 200 ms" in text for text in md_calls)
    assert any("- Cotizaciones: 350 ms" in text and "items: 3" in text for text in md_calls)


def test_render_health_sidebar_with_missing_metrics(monkeypatch):
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
    assert any("⚠️ Error al refrescar" in text and "Token inválido" in text for text in md_calls)
    assert "_Sin consultas registradas._" in md_calls
    assert any("⚠️ API FX con errores" in text and "timeout" in text for text in md_calls)
    assert "_Sin uso de caché registrado._" in md_calls
    assert any(text.startswith("- Portafolio: sin registro") for text in md_calls)
    assert any("- Cotizaciones: s/d" in text and "sin datos" in text for text in md_calls)
