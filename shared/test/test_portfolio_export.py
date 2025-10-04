import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
import sys

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.portfolio_export import (
    PortfolioSnapshotExport,
    build_rankings,
    compute_kpis,
    create_csv_bundle,
    create_excel_workbook,
    write_tables_to_directory,
)


def _snapshot() -> PortfolioSnapshotExport:
    positions = pd.DataFrame(
        [
            {"simbolo": "GGAL", "tipo": "ACCION", "valor_actual": 1200.0, "pl": 200.0},
            {"simbolo": "AL30", "tipo": "BONO", "valor_actual": 800.0, "pl": -50.0},
        ]
    )
    totals = {
        "total_value": 2000.0,
        "total_cost": 1700.0,
        "total_pl": 300.0,
        "total_pl_pct": (300.0 / 1700.0) * 100.0,
        "total_cash": 250.0,
    }
    history = pd.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "total_value": [1800.0, 2000.0],
            "total_cost": [1700.0, 1700.0],
            "total_pl": [100.0, 300.0],
        }
    )
    contrib_symbol = pd.DataFrame(
        {
            "tipo": ["ACCION", "BONO"],
            "simbolo": ["GGAL", "AL30"],
            "valor_actual": [1200.0, 800.0],
            "valor_actual_pct": [60.0, 40.0],
            "pl": [200.0, -50.0],
            "pl_pct": [80.0, -20.0],
        }
    )
    contrib_type = pd.DataFrame(
        {
            "tipo": ["ACCION", "BONO"],
            "valor_actual": [1200.0, 800.0],
            "valor_actual_pct": [60.0, 40.0],
            "pl": [200.0, -50.0],
            "pl_pct": [80.0, -20.0],
        }
    )
    return PortfolioSnapshotExport(
        name="demo",
        generated_at=datetime(2024, 1, 2, 15, 30),
        positions=positions,
        totals=totals,
        history=history,
        contributions_by_symbol=contrib_symbol,
        contributions_by_type=contrib_type,
    )


def test_compute_kpis_returns_metrics() -> None:
    snap = _snapshot()
    df = compute_kpis(snap, metric_keys=["total_value", "positions", "cash_ratio"])
    assert set(df["metric"]) == {"total_value", "positions", "cash_ratio"}
    positions_row = df[df["metric"] == "positions"].iloc[0]
    assert positions_row["raw_value"] == 2.0
    assert positions_row["value"].isdigit()


def test_build_rankings_includes_expected_tables() -> None:
    snap = _snapshot()
    rankings = build_rankings(snap, limit=5)
    keys = {r.key for r in rankings}
    assert {"pl_top", "pl_bottom", "valor_actual"}.issubset(keys)
    top = next(r for r in rankings if r.key == "pl_top")
    assert "Valor" in top.dataframe.columns


def test_create_csv_bundle_contains_expected_members() -> None:
    snap = _snapshot()
    bundle = create_csv_bundle(snap, metric_keys=["total_value"], limit=3)
    from zipfile import ZipFile

    with ZipFile(BytesIO(bundle)) as zf:
        names = set(zf.namelist())
    assert "kpis.csv" in names
    assert "positions.csv" in names
    assert any(name.startswith("ranking_") for name in names)


def test_create_excel_workbook_generates_sheets(monkeypatch: pytest.MonkeyPatch) -> None:
    snap = _snapshot()

    png_bytes = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    )

    monkeypatch.setattr("shared.portfolio_export.fig_to_png_bytes", lambda fig: png_bytes)

    workbook = create_excel_workbook(
        snap,
        metric_keys=["total_value", "positions"],
        chart_keys=["pl_top"],
        limit=3,
    )

    from zipfile import ZipFile

    with ZipFile(BytesIO(workbook)) as zf:
        workbook_xml = zf.read("xl/workbook.xml")
    assert b"KPIs" in workbook_xml
    assert b"Posiciones" in workbook_xml
    assert b"Gr\xc3\xa1ficos" in workbook_xml or "Gr\xc3\xa1ficos".encode("latin1") in workbook_xml


def test_write_tables_to_directory(tmp_path: Path) -> None:
    snap = _snapshot()
    output = write_tables_to_directory(
        snap,
        tmp_path,
        metric_keys=["total_value"],
        include_rankings=False,
        include_history=False,
    )
    assert "kpis" in output and output["kpis"].exists()
    assert output["kpis"].read_text(encoding="utf-8").startswith("snapshot")
