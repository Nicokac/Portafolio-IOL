from __future__ import annotations

"""Helpers to capture runtime environment metadata for diagnostics."""

import csv
import logging
import os
import platform
import sys
import threading
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:  # pragma: no cover - optional dependency, exercised in integration
    import psutil  # type: ignore
except Exception:  # pragma: no cover - defensive fallback when psutil missing
    psutil = None  # type: ignore

try:  # pragma: no cover - Python 3.8 compatibility shim
    from importlib import metadata
except ImportError:  # pragma: no cover - fallback for very old runtimes
    import importlib_metadata as metadata  # type: ignore


analysis_logger = logging.getLogger("analysis")
logger = logging.getLogger(__name__)

_KALEIDO_METRICS_PATH = Path("performance_metrics_15.csv")
_KALEIDO_METRICS_FIELDS = ("kaleido_load_ms",)
_PORTFOLIO_RENDER_COMPLETED_AT: Optional[float] = None
_KALEIDO_LAZY_RECORDED = False
_KALEIDO_LOCK = threading.Lock()


def _safe_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _filter_mapping(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy excluding keys mapped to ``None``."""

    return {key: value for key, value in data.items() if value is not None}


def _collect_cpu_info() -> Dict[str, Any]:
    logical = None
    physical = None
    freq = None

    if psutil is not None:
        logical = _safe_int(psutil.cpu_count(logical=True))
        physical = _safe_int(psutil.cpu_count(logical=False))
        try:
            cpu_freq = psutil.cpu_freq()
        except Exception:  # pragma: no cover - defensive
            cpu_freq = None
        if cpu_freq is not None:
            freq = _safe_float(cpu_freq.current)
    else:
        logical = _safe_int(os.cpu_count())

    info: Dict[str, Any] = {
        "logical_count": logical,
        "physical_count": physical,
    }
    if freq is not None:
        info["frequency_mhz"] = round(freq, 2)

    return _filter_mapping(info)


def _collect_memory_info() -> Dict[str, Any]:
    total_bytes: Optional[int] = None
    available_bytes: Optional[int] = None
    percent_used: Optional[float] = None

    if psutil is not None:
        try:
            memory = psutil.virtual_memory()
        except Exception:  # pragma: no cover - defensive
            memory = None
        if memory is not None:
            total_bytes = _safe_int(memory.total)
            available_bytes = _safe_int(memory.available)
            percent_used = _safe_float(memory.percent)
    else:
        page_size = None
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")  # type: ignore[attr-defined]
        except (AttributeError, ValueError, OSError):  # pragma: no cover - platform specific
            page_size = None
        if page_size is not None:
            try:
                total_pages = os.sysconf("SC_PHYS_PAGES")  # type: ignore[attr-defined]
            except (AttributeError, ValueError, OSError):
                total_pages = None
            if total_pages:
                total_bytes = page_size * total_pages
            try:
                available_pages = os.sysconf("SC_AVPHYS_PAGES")  # type: ignore[attr-defined]
            except (AttributeError, ValueError, OSError):
                available_pages = None
            if available_pages:
                available_bytes = page_size * available_pages

    info: Dict[str, Any] = {}
    if total_bytes is not None:
        info["total_bytes"] = total_bytes
        info["total_mb"] = round(total_bytes / (1024 * 1024), 2)
    if available_bytes is not None:
        info["available_bytes"] = available_bytes
        info["available_mb"] = round(available_bytes / (1024 * 1024), 2)
    if percent_used is not None:
        info["percent_used"] = round(percent_used, 2)

    return info


def _resolve_dependency_version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except Exception:  # pragma: no cover - defensive, handles missing packages
        try:
            module = import_module(package)
        except Exception:
            return None
        return getattr(module, "__version__", None)


def _collect_dependency_versions() -> Dict[str, Optional[str]]:
    packages = ("streamlit", "plotly", "kaleido")
    return {name: _resolve_dependency_version(name) for name in packages}


def _collect_python_info() -> Dict[str, Any]:
    arch = platform.architecture()[0] if hasattr(platform, "architecture") else None
    return _filter_mapping(
        {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "build": " ".join(platform.python_build()),
            "architecture": arch,
            "executable": sys.executable,
        }
    )


def _collect_platform_info() -> Dict[str, Any]:
    return _filter_mapping(
        {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "platform": platform.platform(),
            "node": platform.node(),
        }
    )


def capture_environment_snapshot() -> Dict[str, Any]:
    """Capture and log a snapshot of the runtime environment."""

    snapshot: Dict[str, Any] = {
        "python": _collect_python_info(),
        "platform": _collect_platform_info(),
        "cpu": _collect_cpu_info(),
        "memory": _collect_memory_info(),
        "dependencies": _collect_dependency_versions(),
    }

    metrics: Dict[str, Any] = {}
    cpu_info = snapshot.get("cpu", {})
    if isinstance(cpu_info, Mapping):
        logical = cpu_info.get("logical_count")
        if isinstance(logical, (int, float)):
            metrics["cpu_logical"] = logical
    memory_info = snapshot.get("memory", {})
    if isinstance(memory_info, Mapping):
        total_mb = memory_info.get("total_mb")
        if isinstance(total_mb, (int, float)):
            metrics["memory_total_mb"] = total_mb

    analysis_logger.info(
        "env.snapshot captured",
        extra={
            "analysis": {
                "event": "env.snapshot",
                "latest": snapshot,
                "metrics": metrics,
            }
        },
    )

    return snapshot


def mark_portfolio_ui_render_complete(*, timestamp: Optional[float] = None) -> None:
    """Record the earliest timestamp when the portfolio UI finished rendering."""

    global _PORTFOLIO_RENDER_COMPLETED_AT

    ts = timestamp if timestamp is not None else time.perf_counter()
    with _KALEIDO_LOCK:
        if _PORTFOLIO_RENDER_COMPLETED_AT is None:
            _PORTFOLIO_RENDER_COMPLETED_AT = ts


def _append_kaleido_metric(duration_ms: float) -> None:
    try:
        safe_duration = max(float(duration_ms), 0.0)
    except Exception:
        safe_duration = 0.0

    payload = {"kaleido_load_ms": f"{safe_duration:.2f}"}

    try:
        _KALEIDO_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_exists = _KALEIDO_METRICS_PATH.exists()
        with _KALEIDO_METRICS_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_KALEIDO_METRICS_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(payload)
    except Exception:  # pragma: no cover - best effort logging
        logger.debug(
            "No se pudo actualizar %s con la métrica de Kaleido",
            _KALEIDO_METRICS_PATH,
            exc_info=True,
        )


def record_kaleido_lazy_load(duration_ms: float, *, completed_at: Optional[float] = None) -> None:
    """Persist telemetry about the Kaleido import once it happens lazily."""

    global _KALEIDO_LAZY_RECORDED

    with _KALEIDO_LOCK:
        if _KALEIDO_LAZY_RECORDED:
            return
        _KALEIDO_LAZY_RECORDED = True

    timestamp = completed_at if completed_at is not None else time.perf_counter()
    if _PORTFOLIO_RENDER_COMPLETED_AT is not None:
        delay_ms = max((timestamp - _PORTFOLIO_RENDER_COMPLETED_AT) * 1000.0, 0.0)
        logger.debug(
            "Kaleido lazy init triggered %.2f ms después del render del portafolio",
            delay_ms,
        )

    _append_kaleido_metric(duration_ms)

    load_seconds = max(float(duration_ms), 0.0) / 1000.0
    logger.info(
        "Kaleido initialized lazily after UI render (load=%.1fs)",
        load_seconds,
    )


__all__ = [
    "capture_environment_snapshot",
    "mark_portfolio_ui_render_complete",
    "record_kaleido_lazy_load",
]
