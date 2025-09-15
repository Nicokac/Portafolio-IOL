from types import SimpleNamespace
from unittest.mock import MagicMock

import ui.health_sidebar as health_sidebar


def _build_sidebar_mock():
    return SimpleNamespace(
        header=MagicMock(),
        caption=MagicMock(),
        markdown=MagicMock(),
    )


def test_render_health_sidebar_with_metrics(monkeypatch):
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
        "quotes": {"elapsed_ms": 350.0, "source": "fallback", "count": 3, "ts": 6},
    }
    sidebar_mock = _build_sidebar_mock()
    monkeypatch.setattr(health_sidebar, "st", SimpleNamespace(sidebar=sidebar_mock))
    monkeypatch.setattr(health_sidebar, "get_health_metrics", lambda: metrics)

    health_sidebar.render_health_sidebar()

    md_calls = [call.args[0] for call in sidebar_mock.markdown.call_args_list]
    assert any("#### 🔐 Conexión IOL" in text for text in md_calls)
    assert any("Tokens OK" in text for text in md_calls)
    assert any("Fallback local" in text for text in md_calls)
    assert any("API FX OK" in text for text in md_calls)
    assert any("Uso de caché" in text for text in md_calls)
    assert any("Portafolio" in text and "200" in text for text in md_calls)
    assert any("Cotizaciones" in text and "items: 3" in text for text in md_calls)


def test_render_health_sidebar_without_metrics(monkeypatch):
    metrics = {
        "iol_refresh": None,
        "yfinance": None,
        "fx_api": None,
        "fx_cache": None,
        "portfolio": None,
        "quotes": None,
    }
    sidebar_mock = _build_sidebar_mock()
    monkeypatch.setattr(health_sidebar, "st", SimpleNamespace(sidebar=sidebar_mock))
    monkeypatch.setattr(health_sidebar, "get_health_metrics", lambda: metrics)

    health_sidebar.render_health_sidebar()

    md_calls = [call.args[0] for call in sidebar_mock.markdown.call_args_list]
    assert "_Sin actividad registrada._" in md_calls
    assert "_Sin consultas registradas._" in md_calls
    assert "_Sin llamadas a la API FX._" in md_calls
    assert "_Sin uso de caché registrado._" in md_calls
    assert any(text.startswith("- Portafolio: sin registro") for text in md_calls)
    assert any(text.startswith("- Cotizaciones: sin registro") for text in md_calls)
