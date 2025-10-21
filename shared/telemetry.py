"""Utilities for writing structured telemetry to CSV files."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

try:  # pragma: no cover - optional in certain tests
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - defensive fallback
    st = None  # type: ignore

from shared.version import __build_signature__, __version__

logger = logging.getLogger(__name__)


_METRIC_COLUMNS = (
    "timestamp",
    "metric_name",
    "duration_ms",
    "status",
    "context",
)

_TELEMETRY_DEBOUNCE_SECONDS = 0.3
_LAST_PHASE_LOG: dict[str, float] = {}


def is_hydration_locked() -> bool:
    """Return whether telemetry should be deferred until hydration completes."""

    if st is None:
        return False
    state = getattr(st, "session_state", None)
    if state is None:
        return False
    try:
        return bool(state.get("_hydration_lock"))
    except Exception:  # pragma: no cover - defensive safeguard
        return False


@dataclass(frozen=True)
class TelemetryRow:
    """Structured telemetry data written to CSV logs."""

    metric_name: str
    duration_ms: float | None = None
    status: str = "ok"
    context: Mapping[str, object] | None = None

    def as_dict(self) -> dict[str, str]:
        timestamp = datetime.now(timezone.utc).isoformat()
        payload: dict[str, str] = {
            "timestamp": timestamp,
            "metric_name": str(self.metric_name),
            "duration_ms": _format_milliseconds(self.duration_ms),
            "status": _coerce_status(self.status),
            "context": _serialize_context(self.context),
        }
        return payload


def _format_milliseconds(value: float | None) -> str:
    if value is None:
        return ""
    try:
        coerced = max(float(value), 0.0)
    except (TypeError, ValueError):
        return ""
    return f"{coerced:.2f}"


def _coerce_status(value: object | None) -> str:
    if value is None:
        return "ok"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or "ok"
    if isinstance(value, bool):
        return "ok" if value else "error"
    try:
        return str(value)
    except Exception:  # pragma: no cover - defensive safeguard
        return "ok"


def _serialize_context(context: Mapping[str, object] | None) -> str:
    if not context:
        return "{}"
    normalized = {k: _normalize_context_value(v) for k, v in context.items() if k}
    try:
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):  # pragma: no cover - defensive fallback
        fallback = {str(k): _fallback_string(v) for k, v in normalized.items()}
        return json.dumps(fallback, ensure_ascii=False, sort_keys=True)


def _normalize_context_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(k): _normalize_context_value(v) for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_normalize_context_value(v) for v in value]
    try:
        return json.loads(json.dumps(value, default=_fallback_string))
    except Exception:  # pragma: no cover - defensive fallback
        return _fallback_string(value)


def _fallback_string(value: object) -> str:
    try:
        return str(value)
    except Exception:  # pragma: no cover - defensive safeguard
        return ""


def _ensure_schema(path: Path) -> bool:
    """Ensure the telemetry file at ``path`` uses the expected header."""

    if not path.exists():
        return False
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
    except Exception:  # pragma: no cover - defensive safeguard
        header = None
    expected = list(_METRIC_COLUMNS)
    if header == expected:
        return True
    backup = path.with_suffix(path.suffix + ".legacy")
    try:
        if backup.exists():
            backup.unlink()
        path.rename(backup)
        logger.warning(
            "Telemetry schema changed for %s; previous data moved to %s", path, backup
        )
    except OSError:  # pragma: no cover - fallback when rename fails
        path.unlink(missing_ok=True)
    return False


def _merge_context(
    target: MutableMapping[str, object], source: Mapping[str, object] | None
) -> None:
    if not source:
        return
    for key, value in source.items():
        if key is None:
            continue
        target[str(key)] = value


def _normalize_metric_name(metric_name: object | None, phase: object | None) -> str:
    candidate = metric_name if metric_name not in (None, "") else phase
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    if candidate is None:
        return "unknown"
    try:
        text = str(candidate).strip()
    except Exception:  # pragma: no cover - defensive safeguard
        text = "unknown"
    return text or "unknown"


def _normalize_duration(
    duration_ms: float | None, elapsed_s: float | None
) -> float | None:
    if duration_ms is not None:
        try:
            return max(float(duration_ms), 0.0)
        except (TypeError, ValueError):
            return None
    if elapsed_s is None:
        return None
    try:
        return max(float(elapsed_s), 0.0) * 1000.0
    except (TypeError, ValueError):
        return None


def log_telemetry(
    files: Iterable[Path | str],
    *,
    metric_name: str | None = None,
    phase: str | None = None,
    duration_ms: float | None = None,
    elapsed_s: float | None = None,
    status: object | None = "ok",
    dataset_hash: str | None = None,
    memo_hit_ratio: float | None = None,
    pipeline_cache_hit_ratio: float | None = None,
    subbatch_avg_s: float | None = None,
    ui_total_load_ms: float | None = None,
    context: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
    **legacy_context: object,
) -> None:
    """Append a telemetry row to the provided CSV files using a canonical schema."""

    metric = _normalize_metric_name(metric_name, phase)
    phase_key = metric

    if is_hydration_locked():
        logger.debug("Telemetry locked; skipping metric %s", phase_key)
        return

    now = time.monotonic()
    last_logged = _LAST_PHASE_LOG.get(phase_key)
    if last_logged is not None and now - last_logged < _TELEMETRY_DEBOUNCE_SECONDS:
        logger.debug(
            "Telemetry debounce skipped metric %s (%.3f s since last)",
            phase_key,
            now - last_logged,
        )
        return
    _LAST_PHASE_LOG[phase_key] = now

    merged_context: dict[str, object] = {}
    if dataset_hash is not None:
        merged_context["dataset_hash"] = dataset_hash
    if memo_hit_ratio is not None:
        merged_context["memo_hit_ratio"] = memo_hit_ratio
    if pipeline_cache_hit_ratio is not None:
        merged_context["pipeline_cache_hit_ratio"] = pipeline_cache_hit_ratio
    if subbatch_avg_s is not None:
        merged_context["subbatch_avg_s"] = subbatch_avg_s
    if ui_total_load_ms is not None:
        merged_context["ui_total_load_ms"] = ui_total_load_ms

    _merge_context(merged_context, extra)
    _merge_context(merged_context, context)
    _merge_context(merged_context, legacy_context)

    merged_context.setdefault("app_version", __version__)
    merged_context.setdefault("build_signature", __build_signature__)

    row = TelemetryRow(
        metric_name=metric,
        duration_ms=_normalize_duration(duration_ms, elapsed_s),
        status=status,
        context=merged_context,
    ).as_dict()

    for file_path in files:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = _ensure_schema(path)
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_METRIC_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        logger.info(
            "Telemetry logged to %s metric=%s status=%s", path, metric, row["status"]
        )


DEFAULT_TELEMETRY_FILES: tuple[Path, Path] = (
    Path("performance_metrics_14.csv"),
    Path("performance_metrics_15.csv"),
)


def log_default_telemetry(**kwargs) -> None:
    """Helper that logs telemetry to the default metric files."""

    log_telemetry(DEFAULT_TELEMETRY_FILES, **kwargs)


def log(event_name: str, **kwargs: object) -> None:
    """Emit a simple structured telemetry event to the logger."""

    logger.info("[Telemetry] %s %s", event_name, kwargs)


__all__ = [
    "DEFAULT_TELEMETRY_FILES",
    "log",
    "is_hydration_locked",
    "log_default_telemetry",
    "log_telemetry",
]
