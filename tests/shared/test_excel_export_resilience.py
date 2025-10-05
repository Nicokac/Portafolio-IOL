"""Regression tests for Excel export resilience when chart images are missing."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared import portfolio_export as export


def _snapshot() -> export.PortfolioSnapshotExport:
    return export.PortfolioSnapshotExport(
        name="demo",
        generated_at=None,
        positions=pd.DataFrame([{"simbolo": "AAA", "pl": 10.0, "valor_actual": 100.0}]),
        totals={"total_value": 100.0},
        history=pd.DataFrame(),
        contributions_by_symbol=pd.DataFrame(),
        contributions_by_type=pd.DataFrame(),
    )


def _patch_export_dependencies(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    chart_keys = ["timeline", "composition"]

    def fake_assemble_tables(*_args, **_kwargs):
        return {"kpis": pd.DataFrame([{"metric": "dummy", "value": 1}])}

    def fake_build_chart_figures(
        _snapshot: export.PortfolioSnapshotExport,
        keys: list[str],
        *,
        limit: int = 10,
    ) -> dict[str, go.Figure]:
        return {key: go.Figure() for key in keys}

    monkeypatch.setattr(export, "assemble_tables", fake_assemble_tables)
    monkeypatch.setattr(export, "build_chart_figures", fake_build_chart_figures)

    return chart_keys


@pytest.mark.parametrize("png_bytes", [None, b""])
def test_excel_export_skips_missing_chart_images(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, png_bytes: bytes | None
) -> None:
    chart_keys = _patch_export_dependencies(monkeypatch)
    monkeypatch.setattr(export, "fig_to_png_bytes", lambda _fig: png_bytes)

    caplog.set_level(logging.WARNING, logger=export.logger.name)

    workbook_bytes = export.create_excel_workbook(_snapshot(), chart_keys=chart_keys)

    assert workbook_bytes
    assert len(workbook_bytes) > 0

    for key in chart_keys:
        assert any(
            "⛔ Imagen omitida" in record.message and key in record.message
            for record in caplog.records
        )
