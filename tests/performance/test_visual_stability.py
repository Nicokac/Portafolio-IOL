"""Performance regression tests for visual telemetry stability."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from shared.telemetry import log_telemetry


def test_visual_metrics_under_threshold(tmp_path: Path) -> None:
    csv_path = tmp_path / "visual_metrics.csv"

    log_telemetry(
        [csv_path],
        phase="startup.render_portfolio_complete",
        elapsed_s=6.0,
        ui_total_load_ms=9876.5,
        extra={
            "ui_first_paint_ms": 420.4,
            "lazy_loaded_component": "chart",
            "lazy_load_ms": 215.0,
        },
    )

    log_telemetry(
        [csv_path],
        phase="portfolio.lazy_component",
        elapsed_s=0.3,
        extra={
            "lazy_loaded_component": "table",
            "lazy_load_ms": 120.0,
        },
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2

    ui_row = rows[0]
    assert ui_row["metric_name"] == "startup.render_portfolio_complete"
    ui_context = json.loads(ui_row["context"])
    assert ui_context.get("lazy_loaded_component") == "chart"
    assert float(ui_context["ui_total_load_ms"]) < 10_000
    assert float(ui_context["ui_first_paint_ms"]) < 10_000
    assert float(ui_context["lazy_load_ms"]) >= 0.0

    lazy_row = rows[1]
    assert lazy_row["metric_name"] == "portfolio.lazy_component"
    lazy_context = json.loads(lazy_row["context"])
    assert lazy_context.get("lazy_loaded_component") == "table"
    assert float(lazy_context["lazy_load_ms"]) >= 0.0
    assert "ui_total_load_ms" not in lazy_context
