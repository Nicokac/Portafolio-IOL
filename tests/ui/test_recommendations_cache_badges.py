from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture()
def recommendations_tab(streamlit_stub, monkeypatch: pytest.MonkeyPatch):
    import ui.tabs.recommendations as recommendations_module

    module = importlib.reload(recommendations_module)
    monkeypatch.setattr(module, "st", streamlit_stub)
    monkeypatch.setattr(
        module,
        "build_correlation_figure",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    streamlit_stub.reset()
    return module


@pytest.mark.parametrize(
    "ratio, expected_state",
    [
        (0.8, "green"),
        (0.5, "yellow"),
        (0.2, "red"),
    ],
)
def test_render_cache_status_uses_threshold_colors(recommendations_tab, streamlit_stub, ratio, expected_state) -> None:
    cache_stats = {
        "hit_ratio": ratio,
        "remaining_ttl": 180.0,
        "hits": 8,
        "misses": 2,
        "last_updated": "2024-05-01 10:00:00",
    }

    color = recommendations_tab._render_cache_status(cache_stats)

    assert color == expected_state
    status_entries = streamlit_stub.get_records("status")
    assert status_entries, "Expected cache status to render"
    assert status_entries[-1]["state"] == expected_state
    assert "Cache:" in status_entries[-1]["label"]


def _sample_adaptive_payload() -> dict[str, object]:
    summary = {
        "beta_mean": 0.5,
        "correlation_mean": 0.25,
        "sector_dispersion": 0.12,
        "beta_shift_avg": 0.08,
        "mae": 0.01,
        "rmse": 0.02,
        "bias": 0.005,
        "raw_mae": 0.015,
        "raw_rmse": 0.03,
        "raw_bias": 0.007,
    }
    cache_metadata = {"hit_ratio": 0.9, "last_updated": "2024-05-01"}
    return {
        "summary": summary,
        "beta_shift": pd.Series([0.1, 0.2, 0.15]),
        "historical_correlation": pd.DataFrame({"A": [0.1], "B": [0.2]}),
        "rolling_correlation": pd.DataFrame({"A": [0.1], "B": [0.2]}),
        "correlation_matrix": pd.DataFrame({"A": [0.1], "B": [0.2]}),
        "cache_metadata": cache_metadata,
    }


def test_export_adaptive_report_success_triggers_toasts(
    recommendations_tab, streamlit_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    streamlit_stub.reset()
    streamlit_stub.set_button_result("export_adaptive_report", True)

    report_path = Path("/tmp/adaptive-report.html")
    monkeypatch.setattr(
        recommendations_tab,
        "export_adaptive_report",
        lambda payload: report_path,
    )

    recommendations_tab._render_correlation_tab(_sample_adaptive_payload())

    toasts = [entry["message"] for entry in streamlit_stub.get_records("toast")]
    assert toasts[:2] == [
        "Generando reporte adaptativo...",
        "✅ Reporte generado",
    ]


def test_export_adaptive_report_error_triggers_failure_toast(
    recommendations_tab, streamlit_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    streamlit_stub.reset()
    streamlit_stub.set_button_result("export_adaptive_report", True)

    def _raise_error(_payload: dict[str, object]) -> Path:  # pragma: no cover - helper
        raise RuntimeError("boom")

    monkeypatch.setattr(recommendations_tab, "export_adaptive_report", _raise_error)

    recommendations_tab._render_correlation_tab(_sample_adaptive_payload())

    toasts = [entry["message"] for entry in streamlit_stub.get_records("toast")]
    assert toasts[:2] == [
        "Generando reporte adaptativo...",
        "❌ Error al exportar reporte",
    ]
    errors = [entry["text"] for entry in streamlit_stub.get_records("error")]
    assert any("No se pudo exportar" in message for message in errors)
