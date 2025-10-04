"""Integration test covering snapshot persistence and health telemetry."""

from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from domain.models import Controls
from services.portfolio_view import PortfolioViewModelService
from shared import export as export_tools


def _build_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simbolo": ["AL30", "GGAL"],
            "mercado": ["BCBA", "BCBA"],
            "cantidad": [10, 5],
        }
    )


def _portfolio_frame(value_scale: float) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "simbolo": ["AL30", "GGAL"],
            "mercado": ["BCBA", "BCBA"],
            "tipo": ["Bonos", "Acciones"],
            "valor_actual": [120.0, 200.0],
            "costo": [90.0, 150.0],
            "pl": [30.0, 50.0],
            "pl_d": [5.0, 10.0],
        }
    )
    scaled = base.copy()
    for column in ("valor_actual", "costo", "pl", "pl_d"):
        scaled[column] = base[column] * value_scale
    return scaled


def test_snapshot_export_and_health_flow(monkeypatch, streamlit_stub) -> None:
    import services.health as health_service
    import ui.health_sidebar as health_sidebar

    health_service = importlib.reload(health_service)
    health_sidebar = importlib.reload(health_sidebar)

    positions = _build_positions()
    view_outputs = [_portfolio_frame(1.0), _portfolio_frame(1.1)]

    def fake_apply_filters(df, controls, cli, psvc):  # noqa: ANN001 - signature mimic
        return view_outputs.pop(0)

    monkeypatch.setattr("services.portfolio_view._apply_filters", fake_apply_filters)

    service = PortfolioViewModelService()
    controls = Controls()

    snapshot_a = service.get_portfolio_view(
        positions, controls, cli=SimpleNamespace(), psvc=SimpleNamespace()
    )

    service.invalidate_filters("refresh")

    snapshot_b = service.get_portfolio_view(
        positions, controls, cli=SimpleNamespace(), psvc=SimpleNamespace()
    )

    persisted: list[dict[str, object]] = []
    for snapshot in (snapshot_a, snapshot_b):
        persisted.append(
            {
                "value": snapshot.totals.total_value,
                "positions_csv": export_tools.df_to_csv_bytes(snapshot.df_view),
                "history_csv": export_tools.df_to_csv_bytes(snapshot.historical_total),
            }
        )

    assert len(persisted) == 2
    assert persisted[0]["value"] != persisted[1]["value"]

    history_latest = pd.read_csv(io.BytesIO(persisted[1]["history_csv"]))
    assert len(history_latest) == 2
    assert history_latest["total_value"].iloc[-1] > history_latest["total_value"].iloc[0]

    health_service.record_iol_refresh(True, detail="tokens ok")
    health_service.record_yfinance_usage(
        "fallback", detail="cache", status="fallback", fallback=True
    )
    health_service.record_fx_api_response(error="timeout", elapsed_ms=812.0)
    health_service.record_fx_cache_usage("hit", age=33.0)
    health_service.record_macro_api_usage(
        attempts=[
            {"provider": "fred", "status": "success", "elapsed_ms": 120.0, "ts": 1.0},
            {
                "provider": "worldbank",
                "status": "error",
                "elapsed_ms": 320.0,
                "fallback": True,
                "detail": "missing series",
                "ts": 2.0,
                "missing_series": ["EG.USE.PCAP.KG.OE"],
            },
        ],
        metrics={
            "providers": {
                "fred": {
                    "label": "FRED",
                    "count": 3,
                    "status_counts": {"success": 3},
                    "latency_buckets": {"counts": {"fast": 3}},
                    "latest": {"status": "success", "elapsed_ms": 120.0, "ts": 1.0},
                },
                "worldbank": {
                    "label": "World Bank",
                    "count": 2,
                    "status_counts": {"success": 1, "error": 1},
                    "fallback_count": 1,
                    "latency_buckets": {"counts": {"medium": 2}},
                    "latest": {
                        "status": "error",
                        "elapsed_ms": 320.0,
                        "fallback": True,
                        "ts": 2.0,
                        "detail": "missing series",
                    },
                },
            },
            "overall": {
                "count": 5,
                "status_counts": {"success": 4, "error": 1},
                "fallback_count": 1,
                "latency_buckets": {"counts": {"fast": 3, "medium": 2}},
            },
        },
        latest={
            "provider": "worldbank",
            "status": "error",
            "elapsed_ms": 320.0,
            "fallback": True,
            "ts": 2.0,
            "detail": "missing series",
        },
    )
    health_service.record_opportunities_report(
        mode="hit",
        elapsed_ms=180.0,
        cached_elapsed_ms=240.0,
        universe_initial=120,
        universe_final=60,
        discard_ratio=0.5,
        highlighted_sectors=["Energy", "Utilities"],
        counts_by_origin={"nyse": 30, "bcs": 30},
    )
    health_service.record_portfolio_load(123.0, source="api", detail="fresh")
    health_service.record_quote_load(345.0, source="fallback", count=9)

    metrics = health_service.get_health_metrics()

    macro = metrics.get("macro_api") or {}
    worldbank = macro.get("providers", {}).get("worldbank", {})
    assert worldbank.get("fallback_count") == 1
    assert worldbank.get("status_counts", {}).get("error") == 1
    overall = macro.get("overall", {})
    assert overall.get("fallback_count") == 1
    assert "latency_buckets" in overall

    opportunities_stats = metrics.get("opportunities_stats") or {}
    assert opportunities_stats.get("hit_ratio") == pytest.approx(1.0)

    class _StaticTimeProvider:
        @staticmethod
        def from_timestamp(ts):  # noqa: ANN001 - streamlit compatibility
            return SimpleNamespace(text="2024-01-02 03:04:05")

    monkeypatch.setattr(health_sidebar, "TimeProvider", _StaticTimeProvider)

    health_sidebar.render_health_sidebar()

    sidebar_lines = streamlit_stub.sidebar.markdowns
    assert any("📊 Totales macro" in line for line in sidebar_lines)
    assert any("Fallback" in line for line in sidebar_lines)
    assert any("Portafolio" in line and "123" in line for line in sidebar_lines)
    assert any("Cotizaciones" in line and "345" in line for line in sidebar_lines)
