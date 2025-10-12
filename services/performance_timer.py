from __future__ import annotations

"""Lightweight runtime performance instrumentation helpers."""

import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator

_LOG_ENV_NAME = "PERFORMANCE_LOG_PATH"
_DISABLE_PSUTIL_ENV = "PERFORMANCE_TIMER_DISABLE_PSUTIL"
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_LOGGER_NAME = "performance"
_LOG_SEGMENT_SEPARATOR = " | "
_MESSAGE_PREFIX = "\u23f1\ufe0f "


def _resolve_log_path() -> Path:
    raw_path = os.getenv(_LOG_ENV_NAME, "performance_metrics.log")
    path = Path(raw_path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Fall back to the working directory if the configured parent is not writable.
        path = Path.cwd() / Path(raw_path).name
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Last resort: keep the resolved path; logging will rely on a NullHandler.
            pass
    return path


LOG_PATH: Path = _resolve_log_path()


def _configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_performance_timer_handler", False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    handler: logging.Handler
    try:
        handler = logging.FileHandler(log_path, encoding="utf-8")
    except Exception:
        handler = logging.NullHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    setattr(handler, "_performance_timer_handler", True)
    logger.addHandler(handler)

    if not any(getattr(h, "_performance_timer_handler", False) for h in logger.handlers):
        logger.addHandler(logging.NullHandler())

    return logger


LOGGER = _configure_logger(LOG_PATH)

_FORCE_DISABLE_PSUTIL = os.getenv(_DISABLE_PSUTIL_ENV, "0").strip().lower() in {"1", "true", "yes"}

try:  # pragma: no cover - depends on runtime environment
    if _FORCE_DISABLE_PSUTIL:
        raise ModuleNotFoundError("psutil disabled via env flag")
    import psutil  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised when psutil is absent
    psutil = None  # type: ignore[assignment]
    PROCESS = None
    CPU_COUNT = os.cpu_count() or 1
    _PSUTIL_AVAILABLE = False
else:  # pragma: no cover - requires psutil at runtime
    try:
        PROCESS = psutil.Process(os.getpid())
    except Exception:
        PROCESS = None
    try:
        CPU_COUNT = psutil.cpu_count() or os.cpu_count() or 1
    except Exception:
        CPU_COUNT = os.cpu_count() or 1
    _PSUTIL_AVAILABLE = PROCESS is not None


if not _PSUTIL_AVAILABLE:
    LOGGER.warning(
        "psutil no disponible; métricas de CPU/RAM deshabilitadas en performance_timer",  # noqa: TRY400
    )


def _parse_percent(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = value.strip().rstrip("%")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _sanitize_key(key: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_:\-]", "", key.strip().lower())
    return normalized or "meta"


def _sanitize_extra(extra: Dict[str, Any] | None) -> Dict[str, str]:
    if not extra:
        return {}
    sanitized: Dict[str, str] = {}
    for key, value in extra.items():
        if value is None:
            continue
        key_str = _sanitize_key(str(key))
        try:
            sanitized[key_str] = str(value)
        except Exception:
            sanitized[key_str] = repr(value)
    return sanitized


def _format_message(label: str, duration: float, details: Dict[str, str]) -> str:
    base = f"{_MESSAGE_PREFIX}{label} completado en {duration:.3f}s"
    if not details:
        return base
    segments = [base]
    for key in sorted(details):
        segments.append(f"{key}={details[key]}")
    return _LOG_SEGMENT_SEPARATOR.join(segments)


def _flush_logger() -> None:
    for handler in LOGGER.handlers:
        try:
            handler.flush()
        except Exception:
            pass


@dataclass(frozen=True)
class PerformanceEntry:
    """Represents a single parsed entry from the performance log."""

    timestamp: str
    label: str
    duration_s: float
    cpu_percent: float | None
    ram_percent: float | None
    extras: Dict[str, str] = field(default_factory=dict)
    raw: str = ""

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "label": self.label,
            "duration_s": self.duration_s,
            "cpu_percent": self.cpu_percent,
            "ram_percent": self.ram_percent,
        }
        if self.extras:
            payload["extras"] = dict(self.extras)
        return payload


def _parse_log_line(line: str) -> PerformanceEntry | None:
    if _MESSAGE_PREFIX not in line:
        return None
    try:
        prefix, message = line.split("]", 1)
    except ValueError:
        return None
    timestamp = prefix.replace("[", " ").strip()
    message = message.lstrip()
    if not message.startswith(_MESSAGE_PREFIX):
        return None
    body = message[len(_MESSAGE_PREFIX) :].strip()
    parts = [segment.strip() for segment in body.split(_LOG_SEGMENT_SEPARATOR) if segment]
    if not parts:
        return None
    main = parts[0]
    if " completado en " not in main:
        return None
    label, duration_part = main.split(" completado en ", 1)
    if not duration_part.endswith("s"):
        return None
    try:
        duration = float(duration_part[:-1])
    except ValueError:
        return None
    metrics: Dict[str, str] = {}
    for segment in parts[1:]:
        if "=" not in segment:
            continue
        key, value = segment.split("=", 1)
        metrics[key.strip()] = value.strip()
    cpu = _parse_percent(metrics.pop("cpu", None))
    ram = _parse_percent(metrics.pop("ram", None))
    return PerformanceEntry(
        timestamp=timestamp,
        label=label.strip(),
        duration_s=duration,
        cpu_percent=cpu,
        ram_percent=ram,
        extras=metrics,
        raw=line.strip(),
    )


def read_recent_entries(limit: int = 200) -> list[PerformanceEntry]:
    """Return the latest performance log entries up to ``limit`` items."""

    if limit <= 0:
        return []
    if not LOG_PATH.exists():
        return []
    try:
        lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    entries: list[PerformanceEntry] = []
    for line in lines[-limit:]:
        entry = _parse_log_line(line)
        if entry is not None:
            entries.append(entry)
    return entries


@contextmanager
def performance_timer(label: str, *, extra: Dict[str, Any] | None = None) -> Iterator[None]:
    """Measure a code block, logging duration (and CPU/RAM when available)."""

    start_time = time.perf_counter()
    cpu_start = None
    if _PSUTIL_AVAILABLE and PROCESS is not None:
        try:
            cpu_start = PROCESS.cpu_times()
        except Exception:
            cpu_start = None
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        cpu_pct: float | None = None
        ram_pct: float | None = None
        if _PSUTIL_AVAILABLE and PROCESS is not None:
            try:
                cpu_end = PROCESS.cpu_times()
            except Exception:
                cpu_end = None
            if cpu_start is not None and cpu_end is not None and elapsed > 0:
                cpu_delta = (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)
                if CPU_COUNT:
                    cpu_pct = max(
                        0.0,
                        (cpu_delta / elapsed) * 100.0 / max(float(CPU_COUNT), 0.0001),
                    )
            try:
                ram_pct = PROCESS.memory_percent()
            except Exception:
                ram_pct = None
        details = _sanitize_extra(extra)
        if cpu_pct is not None:
            details.setdefault("cpu", f"{cpu_pct:.1f}%")
        if ram_pct is not None:
            details.setdefault("ram", f"{ram_pct:.1f}%")
        message = _format_message(label, elapsed, details)
        log_extra = {
            "perf_label": label,
            "perf_duration": elapsed,
            "perf_cpu_percent": cpu_pct,
            "perf_ram_percent": ram_pct,
        }
        if details:
            log_extra["perf_meta"] = dict(details)
        LOGGER.info(message, extra=log_extra)
        _flush_logger()


def track_performance(label: str):
    """Decorator that wraps a callable with :func:`performance_timer`."""

    def _decorator(func):
        from functools import wraps

        @wraps(func)
        def _wrapped(*args, **kwargs):
            with performance_timer(label):
                return func(*args, **kwargs)

        return _wrapped

    return _decorator


__all__ = [
    "LOG_PATH",
    "PerformanceEntry",
    "performance_timer",
    "read_recent_entries",
    "track_performance",
]
