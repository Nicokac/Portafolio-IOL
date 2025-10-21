import csv
from pathlib import Path

import pytest

from tools import benchmark_report


@pytest.fixture()
def tmp_metrics_dir(tmp_path: Path) -> Path:
    baseline_dir = tmp_path / "baseline"
    current_dir = tmp_path / "current"
    baseline_dir.mkdir()
    current_dir.mkdir()

    _write_metrics(
        baseline_dir / "qa_metrics.csv",
        {
            "startup": [120.0, 110.0],
            "cache": [45.0],
            "auth": [30.0, 33.0],
            "ui_render": [210.0, 205.0],
        },
    )

    _write_metrics(
        current_dir / "qa_metrics.csv",
        {
            "startup": [100.0, 95.0],
            "cache": [44.0],
            "auth": [29.0, 32.0],
            "ui_render": [205.0, 198.0],
        },
    )

    return tmp_path


def _write_metrics(path: Path, values: dict[str, list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        for metric, samples in values.items():
            for sample in samples:
                writer.writerow([metric, sample])


def test_load_metrics(tmp_metrics_dir: Path) -> None:
    baseline_dir = tmp_metrics_dir / "baseline"
    metrics = benchmark_report.load_metrics(baseline_dir)

    assert metrics["Startup"] == [120.0, 110.0]
    assert metrics["Cache"] == [45.0]
    assert metrics["Auth"] == [30.0, 33.0]
    assert metrics["UI Render"] == [210.0, 205.0]


def test_summarize_metrics(tmp_metrics_dir: Path, tmp_path: Path) -> None:
    baseline_dir = tmp_metrics_dir / "baseline"
    current_dir = tmp_metrics_dir / "current"

    summaries = benchmark_report.run_report(
        baseline_dir,
        current_dir,
        baseline_label="v0.6.x",
        current_label="v0.7.0",
        markdown_output=tmp_path / "benchmark_report.md",
    )

    summary_by_metric = {summary.name: summary for summary in summaries}

    startup_summary = summary_by_metric["Startup"]
    assert pytest.approx(startup_summary.baseline, rel=1e-4) == 115.0
    assert pytest.approx(startup_summary.current, rel=1e-4) == 97.5
    assert startup_summary.status == "improved"
    assert startup_summary.delta_pct < 0

    auth_summary = summary_by_metric["Auth"]
    assert auth_summary.status in {"improved", "neutral"}

    report_path = tmp_path / "benchmark_report.md"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "# Benchmark Comparison" in content
    assert "Startup" in content
