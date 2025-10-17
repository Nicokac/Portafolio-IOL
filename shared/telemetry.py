"""Utilities for writing structured telemetry to CSV files."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)


_METRIC_COLUMNS = (
    "timestamp",
    "phase",
    "elapsed_s",
    "dataset_hash",
    "memo_hit_ratio",
    "subbatch_avg_s",
    "ui_total_load_ms",
    "tab_name",
    "portfolio_tab_render_s",
    "streamlit_overhead_ms",
    "skeleton_render_ms",
    "ui_first_paint_ms",
    "profile_block_total_ms",
    "incremental_render",
    "ui_partial_update_ms",
    "reused_visual_cache",
    "visual_cache_cleared",
    "lazy_loaded_component",
    "lazy_load_ms",
)


@dataclass(frozen=True)
class TelemetryRow:
    """Structured telemetry data written to CSV logs."""

    phase: str
    elapsed_s: float | None = None
    dataset_hash: str | None = None
    memo_hit_ratio: float | None = None
    subbatch_avg_s: float | None = None
    ui_total_load_ms: float | None = None
    extra: Mapping[str, object] | None = None

    def as_dict(self) -> dict[str, str]:
        timestamp = datetime.now(timezone.utc).isoformat()
        payload: dict[str, str] = {
            "timestamp": timestamp,
            "phase": str(self.phase),
            "elapsed_s": _format_seconds(self.elapsed_s),
            "dataset_hash": self.dataset_hash or "",
            "memo_hit_ratio": _format_ratio(self.memo_hit_ratio),
            "subbatch_avg_s": _format_seconds(self.subbatch_avg_s),
            "ui_total_load_ms": _format_milliseconds(self.ui_total_load_ms),
            "tab_name": "",
            "portfolio_tab_render_s": "",
            "streamlit_overhead_ms": "",
            "skeleton_render_ms": "",
            "ui_first_paint_ms": "",
            "profile_block_total_ms": "",
            "incremental_render": "",
            "ui_partial_update_ms": "",
            "reused_visual_cache": "",
            "visual_cache_cleared": "",
            "lazy_loaded_component": "",
            "lazy_load_ms": "",
        }
        if self.extra:
            for key, value in self.extra.items():
                if key not in payload:
                    continue
                payload[key] = _coerce_value(value)
        return payload


def _format_seconds(value: float | None) -> str:
    if value is None:
        return ""
    try:
        return f"{max(float(value), 0.0):.6f}"
    except (TypeError, ValueError):
        return ""


def _format_milliseconds(value: float | None) -> str:
    if value is None:
        return ""
    try:
        return f"{max(float(value), 0.0):.2f}"
    except (TypeError, ValueError):
        return ""


def _format_ratio(value: float | None) -> str:
    if value is None:
        return ""
    try:
        ratio = max(min(float(value), 1.0), 0.0)
    except (TypeError, ValueError):
        return ""
    return f"{ratio:.3f}"


def _coerce_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
    except (TypeError, ValueError):
        return ""
    return str(value)


def log_telemetry(
    files: Iterable[Path | str],
    *,
    phase: str,
    elapsed_s: float | None = None,
    dataset_hash: str | None = None,
    memo_hit_ratio: float | None = None,
    subbatch_avg_s: float | None = None,
    ui_total_load_ms: float | None = None,
    extra: Mapping[str, object] | None = None,
) -> None:
    """Append a telemetry row to the provided CSV files."""

    row = TelemetryRow(
        phase=phase,
        elapsed_s=elapsed_s,
        dataset_hash=dataset_hash,
        memo_hit_ratio=memo_hit_ratio,
        subbatch_avg_s=subbatch_avg_s,
        ui_total_load_ms=ui_total_load_ms,
        extra=extra,
    ).as_dict()

    for file_path in files:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_METRIC_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        logger.info("Telemetry logged to %s", path)


DEFAULT_TELEMETRY_FILES: tuple[Path, Path] = (
    Path("performance_metrics_14.csv"),
    Path("performance_metrics_15.csv"),
)


def log_default_telemetry(**kwargs) -> None:
    """Helper that logs telemetry to the default metric files."""

    log_telemetry(DEFAULT_TELEMETRY_FILES, **kwargs)
