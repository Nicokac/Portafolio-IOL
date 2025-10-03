from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _render(module, metrics: dict[str, Any]) -> None:
    module.st.session_state["health_metrics"] = metrics
    module.render_health_sidebar()


@pytest.fixture
def health_sidebar(streamlit_stub, monkeypatch: pytest.MonkeyPatch):
    import ui.health_sidebar as health_sidebar_module

    module = importlib.reload(health_sidebar_module)
    monkeypatch.setattr(module, "get_health_metrics", lambda: module.st.session_state.get("health_metrics", {}))
    return module


def test_render_health_sidebar_with_success_metrics(health_sidebar) -> None:
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

    _render(health_sidebar, metrics)

    sidebar = health_sidebar.st.sidebar
    assert sidebar.headers == [f"🩺 Healthcheck (versión {health_sidebar.__version__})"]
    markdown_calls = sidebar.markdowns
    assert any("Conexión IOL" in text for text in markdown_calls)
    assert any("Fallback local" in text for text in markdown_calls)
    assert any("API FX OK" in text for text in markdown_calls)
    assert any("Uso de caché" in text for text in markdown_calls)
    assert any("Portafolio" in text and "200" in text for text in markdown_calls)
    assert any("Cotizaciones" in text and "350" in text for text in markdown_calls)


def test_render_health_sidebar_with_missing_metrics(health_sidebar) -> None:
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

    _render(health_sidebar, metrics)

    markdown_calls = health_sidebar.st.sidebar.markdowns
    assert any("Error al refrescar" in text for text in markdown_calls)
    assert "_Sin consultas registradas._" in markdown_calls
    assert any("API FX con errores" in text for text in markdown_calls)
    assert "_Sin uso de caché registrado._" in markdown_calls
    assert any(text.startswith("- Portafolio: sin registro") for text in markdown_calls)
    assert any("Cotizaciones" in text and "sin datos" in text for text in markdown_calls)


@pytest.fixture
def _dummy_metrics() -> dict[str, Any]:
    return {
        "iol_refresh": {"status": "success", "detail": "OK", "ts": None},
        "yfinance": {"source": "yfinance", "detail": "cache", "ts": None},
        "fx_api": {"status": "error", "error": "boom", "elapsed_ms": 250.5, "ts": None},
        "fx_cache": {"mode": "hit", "age": 12.3, "ts": None},
        "macro_api": {
            "attempts": [
                {
                    "provider": "fred",
                    "provider_label": "FRED",
                    "status": "error",
                    "elapsed_ms": 890.0,
                    "detail": "rate limit",
                    "fallback": False,
                    "ts": None,
                },
                {
                    "provider": "ecb",
                    "provider_label": "ECB",
                    "status": "success",
                    "elapsed_ms": 120.0,
                    "fallback": False,
                    "ts": None,
                },
            ],
            "latest": {
                "provider": "fred",
                "provider_label": "FRED",
                "status": "error",
                "elapsed_ms": 890.0,
                "detail": "rate limit",
                "fallback": False,
                "ts": None,
            },
            "providers": {
                "fred": {
                    "label": "FRED",
                    "count": 5,
                    "status_counts": {"success": 3, "error": 2},
                    "status_ratios": {"success": 0.6, "error": 0.4},
                    "error_count": 2,
                    "error_ratio": 0.4,
                    "fallback_count": 1,
                    "fallback_ratio": 0.2,
                    "latency_buckets": {
                        "counts": {"fast": 3, "medium": 1, "slow": 1},
                        "total": 5,
                        "ratios": {"fast": 0.6, "medium": 0.2, "slow": 0.2},
                    },
                    "latest": {
                        "provider": "fred",
                        "provider_label": "FRED",
                        "status": "error",
                        "elapsed_ms": 890.0,
                        "detail": "rate limit",
                        "fallback": False,
                        "ts": None,
                    },
                },
                "ecb": {
                    "label": "ECB",
                    "count": 2,
                    "status_counts": {"success": 2},
                    "status_ratios": {"success": 1.0},
                    "error_count": 0,
                    "error_ratio": 0.0,
                    "fallback_count": 0,
                    "fallback_ratio": 0.0,
                    "latency_buckets": {
                        "counts": {"fast": 1, "medium": 1},
                        "total": 2,
                        "ratios": {"fast": 0.5, "medium": 0.5},
                    },
                    "latest": {
                        "provider": "ecb",
                        "provider_label": "ECB",
                        "status": "success",
                        "elapsed_ms": 120.0,
                        "detail": None,
                        "fallback": False,
                        "ts": None,
                    },
                },
            },
            "overall": {
                "count": 7,
                "status_counts": {"success": 5, "error": 2},
                "status_ratios": {"success": 5 / 7, "error": 2 / 7},
                "fallback_count": 1,
                "fallback_ratio": 1 / 7,
                "error_count": 2,
                "error_ratio": 2 / 7,
                "latency_buckets": {
                    "counts": {"fast": 4, "medium": 2, "slow": 1},
                    "total": 7,
                    "ratios": {"fast": 4 / 7, "medium": 2 / 7, "slow": 1 / 7},
                },
            },
        },
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
        "opportunities_stats": {
            "elapsed": {"avg": 310.0, "stdev": 90.0, "count": 4},
            "cached_elapsed": {"avg": 260.0, "stdev": 60.0, "count": 3},
            "mode_counts": {"hit": 3, "miss": 1},
            "mode_total": 4,
            "mode_ratios": {"hit": 0.75, "miss": 0.25},
            "hit_ratio": 0.75,
            "improvement": {
                "count": 3,
                "wins": 2,
                "losses": 1,
                "ties": 0,
                "win_ratio": 2 / 3,
                "loss_ratio": 1 / 3,
                "tie_ratio": 0.0,
                "avg_delta_ms": 15.0,
            },
        },
    }


def test_format_helpers_use_shared_formatter(health_sidebar, _dummy_metrics) -> None:
    captured: list[str] = []

    def _fake_format(note: str) -> str:
        captured.append(note)
        return f"formatted::{note}"

    health_sidebar.format_note = _fake_format
    health_sidebar.shared_notes.format_note = _fake_format  # type: ignore[attr-defined]

    fx_lines = list(
        health_sidebar._format_fx_section(
            _dummy_metrics["fx_api"], _dummy_metrics["fx_cache"]
        )
    )
    macro_lines = list(health_sidebar._format_macro_section(_dummy_metrics["macro_api"]))
    latency_lines = list(
        health_sidebar._format_latency_section(
            _dummy_metrics["portfolio"], _dummy_metrics["quotes"]
        )
    )
    opportunities_note = health_sidebar._format_opportunities_status(
        _dummy_metrics["opportunities"],
        _dummy_metrics["opportunities_history"],
        _dummy_metrics["opportunities_stats"],
    )
    history_lines = list(
        health_sidebar._format_opportunities_history(
            reversed(_dummy_metrics["opportunities_history"]),
            _dummy_metrics["opportunities_stats"],
        )
    )

    assert captured  # formatter used
    assert fx_lines and macro_lines and latency_lines and history_lines
    assert any("Historial de intentos" in line for line in macro_lines)
    assert "universo 120→48" in opportunities_note
    assert "descartes 60%" in opportunities_note
    assert "sectores: Energy, Utilities" in opportunities_note
    assert "origen: nyse=30, nasdaq=18" in opportunities_note


def test_format_opportunities_status_handles_partial_metrics(health_sidebar) -> None:
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


def test_format_opportunities_status_includes_trend_from_stats(health_sidebar) -> None:
    data = {
        "mode": "hit",
        "elapsed_ms": 200.0,
        "cached_elapsed_ms": 250.0,
        "ts": None,
    }
    stats = {
        "elapsed": {"avg": 210.0, "stdev": 10.0, "count": 4},
        "mode_counts": {"hit": 3, "miss": 1},
        "mode_total": 4,
        "hit_ratio": 0.75,
        "improvement": {"count": 2, "wins": 1, "win_ratio": 0.5, "avg_delta_ms": 5.0},
    }

    note = health_sidebar._format_opportunities_status(data, stats=stats)

    assert "tendencia" in note
    assert "prom 210" in note
    assert "hits 75%" in note
    assert "mejoras 50%" in note
    assert "Δ̄ +5" in note


def test_format_macro_section_handles_missing_data(health_sidebar) -> None:
    assert health_sidebar._format_macro_section(None) == [
        "_Sin datos macro registrados._"
    ]
    assert health_sidebar._format_macro_section({}) == [
        "_Sin datos macro registrados._"
    ]


def test_format_opportunities_history_shows_deltas(health_sidebar) -> None:
    history = [
        {"mode": "hit", "elapsed_ms": 200.0, "cached_elapsed_ms": 180.0, "ts": None},
        {"mode": "miss", "elapsed_ms": 260.0, "cached_elapsed_ms": None, "ts": None},
    ]
    stats = {"elapsed": {"avg": 220.0, "stdev": 20.0, "count": 2}}

    lines = list(health_sidebar._format_opportunities_history(history, stats))

    assert len(lines) == 1
    assert "Δ prom" in lines[0]
    assert "+" in lines[0]
    assert "-" in lines[0]


def test_format_opportunities_history_handles_missing_stats(health_sidebar) -> None:
    history = [
        {"mode": "hit", "elapsed_ms": None, "cached_elapsed_ms": None, "ts": None}
    ]

    lines = list(health_sidebar._format_opportunities_history(history, {}))

    assert len(lines) == 1
    assert lines[0].count("s/d") >= 2


def test_render_health_sidebar_includes_history_section(health_sidebar, _dummy_metrics) -> None:
    _render(health_sidebar, _dummy_metrics)

    assert "#### 🗂️ Historial de screenings" in health_sidebar.st.sidebar.markdowns
    expected_history_lines = list(
        health_sidebar._format_opportunities_history(
            reversed(_dummy_metrics["opportunities_history"]),
            _dummy_metrics.get("opportunities_stats"),
        )
    )
    for line in expected_history_lines:
        assert line in health_sidebar.st.sidebar.markdowns


def test_record_opportunities_report_rotates_history(monkeypatch: pytest.MonkeyPatch) -> None:
    import services.health as health_service

    module = importlib.reload(health_service)
    monkeypatch.setattr(module, "time", type("_T", (), {"time": lambda: 0.0}))

    total_records = module._OPPORTUNITIES_HISTORY_LIMIT + 2

    for idx in range(total_records):
        module.record_opportunities_report(
            mode="hit" if idx % 2 == 0 else "miss",
            elapsed_ms=100 + idx,
            cached_elapsed_ms=50 + idx,
        )

    metrics = module.get_health_metrics()
    history = metrics["opportunities_history"]
    assert len(history) == module._OPPORTUNITIES_HISTORY_LIMIT
    last_entry = metrics["opportunities"]
    assert history[-1] == last_entry

    stats = metrics["opportunities_stats"]
    assert stats["mode_counts"]["hit"] == (total_records + 1) // 2
    assert stats["mode_counts"]["miss"] == total_records // 2
    assert stats["elapsed"]["count"] == total_records
    assert stats["cached_elapsed"]["count"] == total_records

    improvement = stats.get("improvement") or {}
    assert improvement.get("count") == total_records
    assert improvement.get("losses") == total_records


def test_health_metrics_store_is_mutable_between_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    import services.health as health_service

    module = importlib.reload(health_service)
    module.st.session_state.clear()

    module.record_iol_refresh(True, detail="ok")
    module.record_yfinance_usage("fallback", detail="cache")

    metrics = module.get_health_metrics()
    assert metrics["iol_refresh"]["detail"] == "ok"
    assert metrics["yfinance"]["detail"] == "cache"
