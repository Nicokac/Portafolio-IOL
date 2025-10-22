"""Compare QA benchmark metrics between two builds and emit a Markdown report."""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)

METRIC_ORDER: list[str] = ["Startup", "Cache", "Auth", "UI Render"]

METRIC_ALIASES: Mapping[str, str] = {
    "startup": "Startup",
    "startup_time": "Startup",
    "startup-time": "Startup",
    "cache": "Cache",
    "cache_warmup": "Cache",
    "cache-warmup": "Cache",
    "auth": "Auth",
    "authentication": "Auth",
    "ui_render": "UI Render",
    "ui-render": "UI Render",
    "ui render": "UI Render",
    "ui": "UI Render",
    "ui_rendering": "UI Render",
}


@dataclass
class MetricSummary:
    """Aggregate data comparing baseline and current measurements."""

    name: str
    baseline: float
    current: float
    delta_pct: float | None
    status: str


def resolve_metrics_path(source: Path) -> Path:
    """Return the path to ``qa_metrics.csv`` from the given source."""

    if source.is_dir():
        candidate = source / "qa_metrics.csv"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"qa_metrics.csv not found in directory: {source}")

    if source.name != "qa_metrics.csv":
        raise ValueError(f"Expected qa_metrics.csv but received: {source.name}")

    if not source.exists():
        raise FileNotFoundError(f"qa_metrics.csv not found at: {source}")

    return source


def load_metrics(source: Path) -> dict[str, list[float]]:
    """Load QA metrics from ``qa_metrics.csv``."""

    path = resolve_metrics_path(source)
    metric_values: MutableMapping[str, list[float]] = {
        name: [] for name in METRIC_ORDER
    }

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("qa_metrics.csv must contain a header row")

        value_key = None
        for candidate in ("value", "duration", "duration_ms", "duration_seconds"):
            if candidate in reader.fieldnames:
                value_key = candidate
                break

        if value_key is None:
            raise ValueError(
                "qa_metrics.csv must include one of the following columns: "
                "value, duration, duration_ms, duration_seconds"
            )

        if "metric" not in reader.fieldnames:
            raise ValueError("qa_metrics.csv must include a 'metric' column")

        for row in reader:
            raw_metric = row.get("metric", "").strip().lower()
            if not raw_metric:
                continue
            if raw_metric not in METRIC_ALIASES:
                continue

            canonical = METRIC_ALIASES[raw_metric]
            raw_value = row.get(value_key, "").strip()
            if raw_value == "":
                continue

            try:
                value = float(raw_value)
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError(
                    f"Invalid numeric value '{raw_value}' for metric '{canonical}'"
                ) from exc

            metric_values.setdefault(canonical, []).append(value)

    return dict(metric_values)


def average_metrics(metric_samples: Mapping[str, Iterable[float]]) -> dict[str, float]:
    """Compute the average for each metric, ignoring empty samples."""

    averages: dict[str, float] = {}
    for metric in METRIC_ORDER:
        samples = list(metric_samples.get(metric, []))
        if not samples:
            raise ValueError(f"Metric '{metric}' is missing from qa_metrics.csv")
        averages[metric] = statistics.mean(samples)
    return averages


def compute_delta_pct(baseline: float, current: float) -> float | None:
    """Compute the percentage delta relative to the baseline."""

    if baseline == 0:
        return None
    return ((current - baseline) / baseline) * 100


def summarize_metrics(
    baseline_avgs: Mapping[str, float], current_avgs: Mapping[str, float]
) -> list[MetricSummary]:
    """Create metric summaries comparing baseline and current averages."""

    summaries: list[MetricSummary] = []
    for metric in METRIC_ORDER:
        baseline_value = baseline_avgs[metric]
        current_value = current_avgs[metric]
        delta_pct = compute_delta_pct(baseline_value, current_value)

        if delta_pct is None:
            status = "neutral"
        elif delta_pct < 0:
            status = "improved"
        elif abs(delta_pct) < 1e-9:
            status = "neutral"
        else:
            status = "regression"

        summaries.append(
            MetricSummary(
                name=metric,
                baseline=baseline_value,
                current=current_value,
                delta_pct=delta_pct,
                status=status,
            )
        )

    return summaries


def format_console_table(
    summaries: Iterable[MetricSummary], baseline_label: str, current_label: str
) -> str:
    """Return a console-friendly table of the summary values."""

    header = (
        f"{'Metric':<15}{baseline_label:<15}{current_label:<15}"
        f"{'Δ%':<10}{'Status':<12}"
    )
    lines = [header, "-" * len(header)]

    for summary in summaries:
        if summary.delta_pct is None:
            delta_display = "N/A"
        else:
            delta_display = f"{summary.delta_pct:.2f}%"

        lines.append(
            f"{summary.name:<15}{summary.baseline:<15.2f}{summary.current:<15.2f}"
            f"{delta_display:<10}{summary.status:<12}"
        )

    return "\n".join(lines)


def generate_markdown_report(
    summaries: Iterable[MetricSummary],
    baseline_label: str,
    current_label: str,
    output_path: Path,
) -> None:
    """Create a Markdown report summarizing the benchmark comparison."""

    lines = ["# Benchmark Comparison", ""]
    lines.append(
        "| Metric | {} | {} | Δ% | Status |".format(baseline_label, current_label)
    )
    lines.append("| --- | --- | --- | --- | --- |")

    for summary in summaries:
        if summary.delta_pct is None:
            delta_display = "N/A"
        else:
            delta_display = f"{summary.delta_pct:.2f}%"

        lines.append(
            f"| {summary.name} | {summary.baseline:.2f} | {summary.current:.2f} | "
            f"{delta_display} | {summary.status.capitalize()} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Reporte Markdown escrito en %s", output_path)


def run_report(
    baseline_source: Path,
    current_source: Path,
    baseline_label: str,
    current_label: str,
    markdown_output: Path | None = None,
) -> list[MetricSummary]:
    """Load metrics, compute summaries, and write the Markdown report."""

    LOGGER.info("Cargando métricas base desde %s", baseline_source)
    baseline_metrics = load_metrics(baseline_source)
    LOGGER.info("Cargando métricas actuales desde %s", current_source)
    current_metrics = load_metrics(current_source)

    baseline_avgs = average_metrics(baseline_metrics)
    current_avgs = average_metrics(current_metrics)

    summaries = summarize_metrics(baseline_avgs, current_avgs)

    console_table = format_console_table(summaries, baseline_label, current_label)
    print(console_table)

    if markdown_output is not None:
        generate_markdown_report(
            summaries,
            baseline_label,
            current_label,
            markdown_output,
        )

    return summaries


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare QA performance metrics")
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to the baseline build directory or qa_metrics.csv file",
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to the current build directory or qa_metrics.csv file",
    )
    parser.add_argument(
        "--baseline-label",
        default="v0.6.x",
        help="Display label for the baseline build",
    )
    parser.add_argument(
        "--current-label",
        default="v0.7.0",
        help="Display label for the current build",
    )
    parser.add_argument(
        "--output",
        default="benchmark_report.md",
        help="Path to write the Markdown report (use '' to skip)",
    )

    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    baseline_path = Path(args.baseline)
    current_path = Path(args.current)
    output_arg = args.output.strip()
    markdown_output = Path(output_arg) if output_arg else None

    run_report(
        baseline_source=baseline_path,
        current_source=current_path,
        baseline_label=args.baseline_label,
        current_label=args.current_label,
        markdown_output=markdown_output,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
