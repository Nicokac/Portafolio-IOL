from __future__ import annotations

import json
import zipfile
import xml.etree.ElementTree as ET
from importlib import util
from pathlib import Path
import sys

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "export_analysis.py"
SPEC = util.spec_from_file_location("export_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
export_analysis = util.module_from_spec(SPEC)
sys.modules.setdefault("export_analysis", export_analysis)
SPEC.loader.exec_module(export_analysis)


def _write_sample_snapshot(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "name": "sample",
        "generated_at": "2024-01-02T10:30:00",
        "positions": [
            {
                "simbolo": "AAPL",
                "tipo": "ACCION",
                "pl": 1500.0,
                "valor_actual": 5200.0,
            },
            {
                "simbolo": "MSFT",
                "tipo": "ACCION",
                "pl": -300.0,
                "valor_actual": 3100.0,
            },
        ],
        "totals": {
            "total_value": 8300.0,
            "total_cost": 7000.0,
            "total_pl": 1300.0,
            "total_pl_pct": 18.57,
            "total_cash": 1200.0,
        },
        "history": [
            {
                "timestamp": "2024-01-01T12:00:00",
                "total_value": 8000.0,
                "total_cost": 6800.0,
                "total_pl": 1200.0,
            },
            {
                "timestamp": "2024-01-02T10:30:00",
                "total_value": 8300.0,
                "total_cost": 7000.0,
                "total_pl": 1300.0,
            },
        ],
        "contributions": {
            "by_symbol": [
                {"simbolo": "AAPL", "participacion": 62.65},
                {"simbolo": "MSFT", "participacion": 37.35},
            ],
            "by_type": [
                {"tipo": "ACCION", "participacion": 100.0},
            ],
        },
    }
    path = directory / "sample.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False))
    return path


def test_main_generates_csv_and_zip_exports(tmp_path: Path) -> None:
    snapshots_dir = tmp_path / "snapshots"
    _write_sample_snapshot(snapshots_dir)

    output_dir = tmp_path / "exports"

    exit_code = export_analysis.main(
        [
            "--input",
            str(snapshots_dir),
            "--output",
            str(output_dir),
            "--formats",
            "csv",
        ]
    )

    assert exit_code == 0

    snapshot_output = output_dir / "sample"
    assert snapshot_output.is_dir()

    generated_files = {path.name for path in snapshot_output.glob("*.csv")}
    assert {"kpis.csv", "positions.csv"}.issubset(generated_files)

    kpis_df = pd.read_csv(snapshot_output / "kpis.csv")
    assert "snapshot" in kpis_df.columns
    assert kpis_df.loc[0, "snapshot"] == "sample"

    zip_path = snapshot_output / "analysis.zip"
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path) as zf:
        zip_contents = set(zf.namelist())
    assert "kpis.csv" in zip_contents
    assert "positions.csv" in zip_contents

    summary_path = output_dir / "summary.csv"
    assert summary_path.exists()

    summary_df = pd.read_csv(summary_path)
    assert "snapshot" in summary_df.columns
    assert summary_df.loc[0, "snapshot"] == "sample"
    assert "total_value" in summary_df.columns


def test_main_generates_excel_with_charts(tmp_path: Path) -> None:
    snapshot_path = _write_sample_snapshot(tmp_path / "snapshots")

    output_dir = tmp_path / "exports"
    if not export_analysis.CHART_SPECS:
        pytest.skip("No hay gráficos configurados para la exportación")
    chart_key = export_analysis.CHART_SPECS[0].key

    exit_code = export_analysis.main(
        [
            "--input",
            str(snapshot_path),
            "--output",
            str(output_dir),
            "--format",
            "excel",
            "--charts",
            chart_key,
        ]
    )

    assert exit_code == 0

    summary_path = output_dir / "summary.csv"
    assert summary_path.exists()

    excel_path = output_dir / "sample" / "analysis.xlsx"
    assert excel_path.exists()

    with zipfile.ZipFile(excel_path) as zf:
        workbook_xml = ET.fromstring(zf.read("xl/workbook.xml"))
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    sheets_parent = workbook_xml.find("main:sheets", ns)
    assert sheets_parent is not None
    sheet_names = [sheet.attrib.get("name") for sheet in sheets_parent.findall("main:sheet", ns)]
    assert "KPIs" in sheet_names
    assert "Posiciones" in sheet_names
    assert "Gráficos" in sheet_names
