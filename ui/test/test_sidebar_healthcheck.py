"""Assertions around the health sidebar rendering for error scenarios."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import ui.health_sidebar as sidebar


def _make_sidebar_mock() -> SimpleNamespace:
    return SimpleNamespace(
        header=MagicMock(),
        caption=MagicMock(),
        markdown=MagicMock(),
    )


def test_health_panel_renders_problem_states(monkeypatch):
    metrics = {
        "iol_refresh": {"status": "error", "detail": "token", "ts": 1700000000},
        "yfinance": {"source": "error", "detail": "AAPL", "ts": 1700001000},
        "fx_api": {
            "status": "error",
            "error": "timeout",
            "elapsed_ms": 512.8,
            "ts": 1700002000,
        },
        "fx_cache": {"mode": "miss", "age": 21.4, "ts": 1700003000},
        "portfolio": {
            "elapsed_ms": 99.9,
            "source": "api",
            "detail": "cache-miss",
            "ts": 1700004000,
        },
        "quotes": {
            "elapsed_ms": 77.7,
            "source": "api",
            "count": 12,
            "detail": "partial",
            "ts": 1700005000,
        },
    }
    sidebar_mock = _make_sidebar_mock()
    monkeypatch.setattr(sidebar, "st", SimpleNamespace(sidebar=sidebar_mock))
    monkeypatch.setattr(sidebar, "get_health_metrics", lambda: metrics)

    sidebar.render_health_sidebar()

    rendered = [entry.args[0] for entry in sidebar_mock.markdown.call_args_list]
    assert any("⚠️ Error al refrescar" in text for text in rendered)
    assert any("⚠️ Error o sin datos" in text for text in rendered)
    assert any("⚠️ API FX con errores" in text for text in rendered)
    assert any("🔄 Actualización" in text for text in rendered)
    assert any("cache-miss" in text for text in rendered)
    assert any("partial" in text for text in rendered)
    assert any("items: 12" in text for text in rendered)

