from __future__ import annotations

"""Lightweight runtime performance instrumentation helpers with observability."""

import atexit
import json
import logging
import os
import queue
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import LogRecord
from logging.handlers import QueueHandler, QueueListener, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterator

from shared.settings import app_env, settings

_LOG_ENV_NAME = "PERFORMANCE_LOG_PATH"
_DISABLE_PSUTIL_ENV = "PERFORMANCE_TIMER_DISABLE_PSUTIL"
_JSON_LOG_ENV = "PERFORMANCE_JSON_LOG_PATH"
_JSON_LOG_DEFAULT = Path("logs/performance/structured.log")
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_LOGGER_NAME = "performance"
_LOG_SEGMENT_SEPARATOR = " | "
_MESSAGE_PREFIX = "\u23f1\ufe0f "
_DEFAULT_JSON_BACKUP_DAYS = 14

_VERBOSE_TEXT_LOG = bool(getattr(settings, "PERFORMANCE_VERBOSE_TEXT_LOG", False))
_REDIS_URL = getattr(settings, "REDIS_URL", None)
_ENABLE_PROMETHEUS = bool(getattr(settings, "ENABLE_PROMETHEUS", True))

try:  # pragma: no cover - optional dependency
    from prometheus_client import CollectorRegistry, Gauge, Summary, Counter
except ModuleNotFoundError:  # pragma: no cover - exercised when dependency missing
    CollectorRegistry = None  # type: ignore[assignment]
    Gauge = None  # type: ignore[assignment]
    Summary = None  # type: ignore[assignment]
    Counter = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import redis  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised when dependency missing
    redis = None  # type: ignore[assignment]

_PROMETHEUS_REGISTRY = (
    CollectorRegistry(auto_describe=True) if CollectorRegistry and _ENABLE_PROMETHEUS else None
)

if Summary and Gauge and Counter and _PROMETHEUS_REGISTRY is not None:
    PERFORMANCE_DURATION_SECONDS = Summary(
        "performance_duration_seconds",
        "Execution time for instrumented blocks",
        labelnames=("module", "label", "success"),
        registry=_PROMETHEUS_REGISTRY,
    )
    PERFORMANCE_CPU_PERCENT = Gauge(
        "performance_cpu_percent",
        "CPU percent used by instrumented blocks",
        labelnames=("module", "label", "success"),
        registry=_PROMETHEUS_REGISTRY,
    )
    PERFORMANCE_MEMORY_PERCENT = Gauge(
        "performance_memory_percent",
        "Memory percent used by instrumented blocks",
        labelnames=("module", "label", "success"),
        registry=_PROMETHEUS_REGISTRY,
    )
    UI_TOTAL_LOAD_MS = Gauge(
        "ui_total_load_ms",
        "Total UI load time in milliseconds (from Streamlit startup to full render).",
        registry=_PROMETHEUS_REGISTRY,
    )
    UI_STARTUP_LOAD_MS = Gauge(
        "ui_startup_load_ms",
        "Login render latency in milliseconds before the authenticated UI loads.",
        registry=_PROMETHEUS_REGISTRY,
    )
    PRELOAD_TOTAL_MS = Gauge(
        "preload_total_ms",
        "Total time spent importing scientific libraries after authentication in milliseconds.",
        registry=_PROMETHEUS_REGISTRY,
    )
    PRELOAD_PANDAS_MS = Gauge(
        "preload_pandas_ms",
        "Duration of pandas preload in milliseconds.",
        registry=_PROMETHEUS_REGISTRY,
    )
    PRELOAD_PLOTLY_MS = Gauge(
        "preload_plotly_ms",
        "Duration of plotly preload in milliseconds.",
        registry=_PROMETHEUS_REGISTRY,
    )
    PRELOAD_STATSMODELS_MS = Gauge(
        "preload_statsmodels_ms",
        "Duration of statsmodels preload in milliseconds.",
        registry=_PROMETHEUS_REGISTRY,
    )
    QUOTES_SWR_SERVED_TOTAL = Counter(
        "quotes_swr_served_total",
        "Number of quote batches served via stale-while-revalidate",
        labelnames=("mode",),
        registry=_PROMETHEUS_REGISTRY,
    )
    QUOTES_BATCH_LATENCY_SECONDS = Summary(
        "quotes_batch_latency_seconds",
        "Latency of individual quote refresh batches",
        labelnames=("batch_size", "background"),
        registry=_PROMETHEUS_REGISTRY,
    )
else:  # pragma: no cover - when prometheus_client missing or disabled
    PERFORMANCE_DURATION_SECONDS = None  # type: ignore[assignment]
    PERFORMANCE_CPU_PERCENT = None  # type: ignore[assignment]
    PERFORMANCE_MEMORY_PERCENT = None  # type: ignore[assignment]
    QUOTES_SWR_SERVED_TOTAL = None  # type: ignore[assignment]
    QUOTES_BATCH_LATENCY_SECONDS = None  # type: ignore[assignment]
    UI_TOTAL_LOAD_MS = None  # type: ignore[assignment]
    UI_STARTUP_LOAD_MS = None  # type: ignore[assignment]
    PRELOAD_TOTAL_MS = None  # type: ignore[assignment]
    PRELOAD_PANDAS_MS = None  # type: ignore[assignment]
    PRELOAD_PLOTLY_MS = None  # type: ignore[assignment]
    PRELOAD_STATSMODELS_MS = None  # type: ignore[assignment]


def update_ui_total_load_metric(total_ms: float | int | None) -> None:
    """Propagate UI total load values to the Prometheus gauge if enabled."""

    if UI_TOTAL_LOAD_MS is None:
        return
    try:
        value = float(total_ms) if total_ms is not None else float("nan")
    except Exception:
        value = float("nan")
    UI_TOTAL_LOAD_MS.set(value)


def update_ui_startup_load_metric(total_ms: float | int | None) -> None:
    """Propagate login render timings to the Prometheus gauge if enabled."""

    if UI_STARTUP_LOAD_MS is None:
        return
    try:
        value = float(total_ms) if total_ms is not None else float("nan")
    except Exception:
        value = float("nan")
    UI_STARTUP_LOAD_MS.set(value)


_PRELOAD_LIBRARY_GAUGES: dict[str, Gauge | None] = {
    "pandas": PRELOAD_PANDAS_MS if "PRELOAD_PANDAS_MS" in globals() else None,
    "plotly": PRELOAD_PLOTLY_MS if "PRELOAD_PLOTLY_MS" in globals() else None,
    "statsmodels": PRELOAD_STATSMODELS_MS if "PRELOAD_STATSMODELS_MS" in globals() else None,
}


def update_preload_total_metric(total_ms: float | int | None) -> None:
    """Expose the overall scientific preload duration to Prometheus."""

    gauge = globals().get("PRELOAD_TOTAL_MS")
    if gauge is None:
        return
    try:
        value = float(total_ms) if total_ms is not None else float("nan")
    except Exception:
        value = float("nan")
    gauge.set(value)


def update_preload_library_metric(library: str, duration_ms: float | int | None) -> None:
    """Propagate per-library preload timings to dedicated gauges when available."""

    gauge = _PRELOAD_LIBRARY_GAUGES.get(library)
    if gauge is None:
        return
    try:
        value = float(duration_ms) if duration_ms is not None else float("nan")
    except Exception:
        value = float("nan")
    gauge.set(value)


def init_metrics() -> None:
    """Compatibility shim for lazy initialisation callers."""

    # Import-time side-effects already set up metrics; this function exists for
    # backwards compatibility with callers that expect an explicit hook.
    return None


def _resolve_log_path() -> Path:
    raw_path = os.getenv(_LOG_ENV_NAME, "performance_metrics.log")
    path = Path(raw_path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        path = Path.cwd() / Path(raw_path).name
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    return path


def _resolve_structured_path() -> Path:
    raw_path = os.getenv(_JSON_LOG_ENV, str(_JSON_LOG_DEFAULT))
    path = Path(raw_path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        fallback = Path.cwd() / Path(raw_path).name
        try:
            fallback.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        path = fallback
    return path


LOG_PATH: Path = _resolve_log_path()
STRUCTURED_LOG_PATH: Path = _resolve_structured_path()


class _JsonLineFormatter(logging.Formatter):
    def format(self, record: LogRecord) -> str:
        entry: PerformanceEntry | None = getattr(record, "perf_entry", None)
        if entry is None:
            payload: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": super().format(record),
            }
        else:
            payload = entry.as_dict(include_raw=False)
        return json.dumps(payload, ensure_ascii=False)


class _PerformanceDispatchHandler(logging.Handler):
    """Broadcast structured entries to persistence layers asynchronously."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self._redis_client = None
        self._redis_attempted = False

    def _publish_redis(self, entry: PerformanceEntry) -> None:
        if not _REDIS_URL or redis is None:
            return
        if self._redis_client is None and not self._redis_attempted:
            try:
                self._redis_client = redis.Redis.from_url(_REDIS_URL)
            except Exception:  # pragma: no cover - connection issues depend on runtime
                self._redis_client = None
            finally:
                self._redis_attempted = True
        if self._redis_client is None:
            return
        try:  # pragma: no cover - depends on redis availability
            self._redis_client.xadd(
                "redis_streams:performance",
                {"entry": json.dumps(entry.as_dict(include_raw=False), ensure_ascii=False)},
                maxlen=None,
            )
        except Exception:
            pass

    def _store_sqlite(self, entry: PerformanceEntry) -> None:
        if app_env != "prod":
            return
        try:
            from services.performance_store import store_entry
        except Exception:  # pragma: no cover - defensive import guard
            return
        try:
            store_entry(entry)
        except Exception:
            pass

    def _record_metrics(self, entry: PerformanceEntry) -> None:
        if not _ENABLE_PROMETHEUS:
            return
        if PERFORMANCE_DURATION_SECONDS is None:
            return
        labels = {
            "module": entry.module,
            "label": entry.label,
            "success": "true" if entry.success else "false",
        }
        PERFORMANCE_DURATION_SECONDS.labels(**labels).observe(entry.duration_s)
        if entry.cpu_percent is not None:
            PERFORMANCE_CPU_PERCENT.labels(**labels).set(entry.cpu_percent)
        if entry.ram_percent is not None:
            PERFORMANCE_MEMORY_PERCENT.labels(**labels).set(entry.ram_percent)

    def emit(self, record: LogRecord) -> None:  # pragma: no cover - integration behaviour
        entry: PerformanceEntry | None = getattr(record, "perf_entry", None)
        if entry is None:
            return
        try:
            self._record_metrics(entry)
            self._publish_redis(entry)
            self._store_sqlite(entry)
        except Exception:
            pass


def _build_text_handler(log_path: Path) -> logging.Handler:
    try:
        handler = logging.FileHandler(log_path, encoding="utf-8")
    except Exception:
        handler = logging.NullHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    setattr(handler, "_performance_timer_handler", True)
    return handler


def _build_json_handler(path: Path) -> logging.Handler:
    try:
        handler = TimedRotatingFileHandler(
            path,
            when="midnight",
            backupCount=_DEFAULT_JSON_BACKUP_DAYS,
            encoding="utf-8",
            utc=True,
        )
    except Exception:
        handler = logging.NullHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(_JsonLineFormatter())
    setattr(handler, "_performance_timer_handler", True)
    return handler


_LOG_QUEUE: queue.Queue[LogRecord] | None = None
_LISTENER: QueueListener | None = None
_LISTENER_HANDLERS: list[logging.Handler] = []


def _shutdown_listener() -> None:
    global _LISTENER, _LISTENER_HANDLERS
    if _LISTENER is not None:
        try:
            _LISTENER.stop()
        except Exception:
            pass
        _LISTENER = None
    for handler in _LISTENER_HANDLERS:
        try:
            handler.flush()
        except Exception:
            pass
        try:
            handler.close()
        except Exception:
            pass
    _LISTENER_HANDLERS = []


def _configure_logger(log_path: Path) -> logging.Logger:
    global _LOG_QUEUE, _LISTENER, _LISTENER_HANDLERS
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    _shutdown_listener()

    for handler in list(logger.handlers):
        if getattr(handler, "_performance_timer_handler", False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    _LOG_QUEUE = queue.Queue()
    queue_handler = QueueHandler(_LOG_QUEUE)
    queue_handler.setLevel(logging.INFO)
    setattr(queue_handler, "_performance_timer_handler", True)
    logger.addHandler(queue_handler)

    handlers = [
        _build_text_handler(log_path),
        _build_json_handler(STRUCTURED_LOG_PATH),
        _PerformanceDispatchHandler(),
    ]
    _LISTENER_HANDLERS = handlers
    _LISTENER = QueueListener(
        _LOG_QUEUE, *handlers, respect_handler_level=True
    )
    try:
        _LISTENER.start()
    except Exception:
        pass

    return logger


LOGGER = _configure_logger(LOG_PATH)
atexit.register(_shutdown_listener)

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
        "psutil no disponible; métricas de CPU/RAM deshabilitadas en performance_timer",
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
    base = f"{label} completado en {duration:.3f}s"
    if _VERBOSE_TEXT_LOG:
        base = f"{_MESSAGE_PREFIX}{base}"
    if not details:
        return base
    segments = [base]
    for key in sorted(details):
        segments.append(f"{key}={details[key]}")
    return _LOG_SEGMENT_SEPARATOR.join(segments)


def _resolve_module_name() -> str:
    try:
        frame = sys._getframe(2)
    except (AttributeError, ValueError):  # pragma: no cover - interpreter limitations
        return __name__
    while frame is not None:
        module = frame.f_globals.get("__name__")
        if module and module != __name__ and not module.startswith("contextlib"):
            return str(module)
        frame = frame.f_back
    return __name__


@dataclass
class ProfileBlockResult:
    """Aggregated metrics for an instrumented code block."""

    label: str
    duration_s: float = 0.0
    cpu_percent: float | None = None
    ram_percent: float | None = None
    module: str = __name__
    success: bool = True
    extras: Dict[str, str] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.duration_s * 1000.0


def _capture_cpu_snapshot():
    if not _PSUTIL_AVAILABLE or PROCESS is None:
        return None
    try:
        return PROCESS.cpu_times()
    except Exception:
        return None


def _collect_resource_metrics(cpu_start, elapsed: float) -> tuple[float | None, float | None]:
    cpu_pct: float | None = None
    ram_pct: float | None = None
    if _PSUTIL_AVAILABLE and PROCESS is not None:
        cpu_end = None
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
    return cpu_pct, ram_pct


@dataclass(frozen=True)
class PerformanceEntry:
    """Represents a structured performance telemetry entry."""

    timestamp: str
    label: str
    duration_s: float
    cpu_percent: float | None
    ram_percent: float | None
    extras: Dict[str, str] = field(default_factory=dict)
    module: str = "unknown"
    success: bool = True
    raw: str = ""

    def as_dict(self, *, include_raw: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "label": self.label,
            "duration_s": self.duration_s,
            "cpu_percent": self.cpu_percent,
            "mem_percent": self.ram_percent,
            "module": self.module,
            "success": self.success,
        }
        if self.extras:
            payload["extras"] = dict(self.extras)
        if include_raw and self.raw:
            payload["raw"] = self.raw
        return payload


def _parse_log_line(line: str) -> PerformanceEntry | None:
    prefix_marker = "]"
    if prefix_marker not in line:
        return None
    try:
        prefix, message = line.split(prefix_marker, 1)
    except ValueError:
        return None
    timestamp = prefix.replace("[", " ").strip()
    message = message.lstrip()
    if _MESSAGE_PREFIX in message:
        message = message.replace(_MESSAGE_PREFIX, "", 1).strip()
    parts = [segment.strip() for segment in message.split(_LOG_SEGMENT_SEPARATOR) if segment]
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
    status = metrics.get("status")
    success = status.lower() != "error" if isinstance(status, str) else True
    module_name = metrics.pop("module", None)
    if isinstance(module_name, str):
        module_name = module_name.strip() or None
    module_value = module_name or "unknown"
    return PerformanceEntry(
        timestamp=timestamp,
        label=label.strip(),
        duration_s=duration,
        cpu_percent=cpu,
        ram_percent=ram,
        extras=metrics,
        module=module_value,
        success=success,
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


def _build_entry(
    *,
    label: str,
    duration: float,
    cpu_pct: float | None,
    ram_pct: float | None,
    details: Dict[str, str],
    module: str,
    success: bool,
    raw_message: str,
) -> PerformanceEntry:
    timestamp = datetime.now(timezone.utc).isoformat()
    return PerformanceEntry(
        timestamp=timestamp,
        label=label,
        duration_s=duration,
        cpu_percent=cpu_pct,
        ram_percent=ram_pct,
        extras=details,
        module=module,
        success=success,
        raw=raw_message,
    )


def _log_performance(
    *,
    label: str,
    duration: float,
    cpu_pct: float | None,
    ram_pct: float | None,
    extra: Dict[str, Any] | None,
    success: bool,
    module: str,
) -> PerformanceEntry:
    sanitized = _sanitize_extra(extra)
    if cpu_pct is not None:
        sanitized.setdefault("cpu", f"{cpu_pct:.1f}%")
    if ram_pct is not None:
        sanitized.setdefault("ram", f"{ram_pct:.1f}%")
    sanitized.setdefault("status", "ok" if success else "error")
    log_details = dict(sanitized)
    log_details.setdefault("module", module)
    message = _format_message(label, duration, log_details)
    entry = _build_entry(
        label=label,
        duration=duration,
        cpu_pct=cpu_pct,
        ram_pct=ram_pct,
        details=sanitized,
        module=module,
        success=success,
        raw_message=message,
    )
    log_extra = {
        "perf_label": label,
        "perf_duration": duration,
        "perf_cpu_percent": cpu_pct,
        "perf_ram_percent": ram_pct,
        "perf_meta": dict(sanitized) if sanitized else {},
        "perf_module": module,
        "perf_entry": entry,
    }
    LOGGER.info(message, extra=log_extra)
    return entry


@contextmanager
def performance_timer(label: str, *, extra: Dict[str, Any] | None = None) -> Iterator[None]:
    """Measure a code block, logging duration (and CPU/RAM when available)."""

    with profile_block(label, extra=extra):
        yield


@contextmanager
def profile_block(
    label: str, *, extra: Dict[str, Any] | None = None
) -> Iterator[ProfileBlockResult]:
    """Profile a code block capturing duration, CPU and memory usage."""

    module = _resolve_module_name()
    result = ProfileBlockResult(label=label, module=module)
    start_time = time.perf_counter()
    cpu_start = _capture_cpu_snapshot()
    success = True
    try:
        yield result
    except Exception:
        success = False
        result.success = False
        raise
    finally:
        elapsed = max(time.perf_counter() - start_time, 0.0)
        cpu_pct, ram_pct = _collect_resource_metrics(cpu_start, elapsed)
        result.duration_s = elapsed
        result.cpu_percent = cpu_pct
        result.ram_percent = ram_pct
        result.module = module
        result.success = success
        entry = _log_performance(
            label=label,
            duration=elapsed,
            cpu_pct=cpu_pct,
            ram_pct=ram_pct,
            extra=extra,
            success=success,
            module=module,
        )
        result.extras = dict(entry.extras)


def record_stage(
    label: str,
    *,
    total_ms: float | int,
    status: str = "success",
    extra: Dict[str, Any] | None = None,
) -> None:
    """Log a timing entry without wrapping the measured block."""

    try:
        duration_s = max(float(total_ms) / 1000.0, 0.0)
    except Exception:
        duration_s = 0.0
    metadata: Dict[str, Any] = {}
    if extra:
        metadata.update(extra)
    metadata.setdefault("status", status)
    try:
        metadata.setdefault("total_ms", f"{float(total_ms):.2f}")
    except Exception:
        metadata.setdefault("total_ms", str(total_ms))
    status_value = str(metadata.get("status", ""))
    normalized_status = status_value.strip().lower()
    success = normalized_status not in {"error", "failed", "failure"}
    module = _resolve_module_name()
    _log_performance(
        label=label,
        duration=duration_s,
        cpu_pct=None,
        ram_pct=None,
        extra=metadata,
        success=success,
        module=module,
    )
    if label == "ui_total_load":
        update_ui_total_load_metric(total_ms)


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


PROMETHEUS_REGISTRY = _PROMETHEUS_REGISTRY
PROMETHEUS_ENABLED = _ENABLE_PROMETHEUS and PROMETHEUS_REGISTRY is not None

__all__ = [
    "LOG_PATH",
    "STRUCTURED_LOG_PATH",
    "PerformanceEntry",
    "ProfileBlockResult",
    "PROMETHEUS_REGISTRY",
    "PROMETHEUS_ENABLED",
    "UI_TOTAL_LOAD_MS",
    "UI_STARTUP_LOAD_MS",
    "PRELOAD_TOTAL_MS",
    "PRELOAD_PANDAS_MS",
    "PRELOAD_PLOTLY_MS",
    "PRELOAD_STATSMODELS_MS",
    "profile_block",
    "performance_timer",
    "record_stage",
    "read_recent_entries",
    "track_performance",
    "update_ui_total_load_metric",
    "update_ui_startup_load_metric",
    "update_preload_total_metric",
    "update_preload_library_metric",
    "QUOTES_SWR_SERVED_TOTAL",
    "QUOTES_BATCH_LATENCY_SECONDS",
]
