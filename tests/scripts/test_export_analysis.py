from __future__ import annotations

import base64
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from importlib import util
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "export_analysis.py"
SPEC = util.spec_from_file_location("export_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
export_analysis = util.module_from_spec(SPEC)
sys.modules.setdefault("export_analysis", export_analysis)
SPEC.loader.exec_module(export_analysis)


def _write_sample_snapshot(directory: Path, *, name: str = "sample") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "name": name,
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
    path = directory / f"{name}.json"
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


def test_cli_generates_excel_without_streamlit_cache(tmp_path: Path) -> None:
    snapshots_dir = tmp_path / "snapshots"
    _write_sample_snapshot(snapshots_dir)

    output_dir = tmp_path / "exports"
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.export_analysis",
            "--input",
            str(snapshots_dir),
            "--output",
            str(output_dir),
            "--format",
            "excel",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout

    excel_path = output_dir / "sample" / "analysis.xlsx"
    assert excel_path.exists()


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


def test_main_handles_multiple_snapshots_with_combined_exports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshots_dir = tmp_path / "snapshots"
    names = ["alpha", "beta"]
    for name in names:
        _write_sample_snapshot(snapshots_dir, name=name)

    metrics = [
        "total_value",
        "total_pl",
        "total_pl_pct",
        "total_cash",
        "positions",
        "symbols",
    ]
    chart_keys = [spec.key for spec in export_analysis.CHART_SPECS[:3]]
    if len(chart_keys) < 2:
        pytest.skip("Se requieren al menos dos gráficos para validar la exportación combinada")

    png_bytes = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    )
    monkeypatch.setattr("shared.portfolio_export.fig_to_png_bytes", lambda fig: png_bytes)

    output_dir = tmp_path / "exports"
    exit_code = export_analysis.main(
        [
            "--input",
            str(snapshots_dir),
            "--output",
            str(output_dir),
            "--formats",
            "both",
            "--metrics",
            *metrics,
            "--charts",
            *chart_keys,
        ]
    )

    assert exit_code == 0

    summary_path = output_dir / "summary.csv"
    assert summary_path.exists()
    summary_df = pd.read_csv(summary_path)
    assert set(summary_df["snapshot"]) == set(names)
    for metric in metrics:
        assert metric in summary_df.columns

    for name in names:
        snapshot_dir = output_dir / name
        assert snapshot_dir.is_dir()

        zip_path = snapshot_dir / "analysis.zip"
        assert zip_path.exists()
        with zipfile.ZipFile(zip_path) as zf:
            members = set(zf.namelist())
        assert {"kpis.csv", "positions.csv"}.issubset(members)
        assert any(member.startswith("ranking_") for member in members)

        excel_path = snapshot_dir / "analysis.xlsx"
        assert excel_path.exists()
        with zipfile.ZipFile(excel_path) as zf:
            workbook_files = set(zf.namelist())
            assert "xl/workbook.xml" in workbook_files
            drawing_files = [name for name in workbook_files if name.startswith("xl/drawings/drawing")]
            assert drawing_files
            drawing_xml = ET.fromstring(zf.read(drawing_files[0]))
        ns = {"xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"}
        anchors = drawing_xml.findall("xdr:twoCellAnchor", ns) + drawing_xml.findall("xdr:oneCellAnchor", ns)
        assert len(anchors) >= 2


def test_cli_generates_summary_with_multiple_charts(tmp_path: Path) -> None:
    snapshots_dir = tmp_path / "snapshots"
    names = ["cli_alpha", "cli_beta", "cli_gamma"]
    for name in names:
        _write_sample_snapshot(snapshots_dir, name=name)

    output_dir = tmp_path / "exports"
    metrics = ["total_value", "total_pl", "cash_ratio"]
    chart_keys = [spec.key for spec in export_analysis.CHART_SPECS[:2]]
    if len(chart_keys) < 2:
        pytest.skip("Se requieren al menos dos gráficos para probar la bandera --charts")

    cmd = [
        sys.executable,
        str(MODULE_PATH),
        "--input",
        str(snapshots_dir),
        "--output",
        str(output_dir),
        "--formats",
        "excel",
        "--metrics",
        *metrics,
        "--charts",
        *chart_keys,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    summary_path = output_dir / "summary.csv"
    assert summary_path.exists()
    summary_df = pd.read_csv(summary_path)
    assert len(summary_df) == len(names)
    assert set(summary_df["snapshot"]) == set(names)
    for metric in metrics:
        assert metric in summary_df.columns

    for name in names:
        excel_path = output_dir / name / "analysis.xlsx"
        assert excel_path.exists()
