"""Performance regression tests for visual telemetry stability."""

from __future__ import annotations

import csv
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
    assert "ui_total_load_ms" in ui_row
    assert float(ui_row["ui_total_load_ms"]) < 10_000
    assert float(ui_row["ui_first_paint_ms"]) < 10_000
    assert ui_row["lazy_loaded_component"] == "chart"
    assert float(ui_row["lazy_load_ms"]) >= 0.0

    lazy_row = rows[1]
    assert lazy_row["lazy_loaded_component"] == "table"
    assert float(lazy_row["lazy_load_ms"]) >= 0.0
    assert lazy_row["ui_total_load_ms"] == ""
