from __future__ import annotations

import base64
import json
import logging
import os
import platform
import statistics
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Deque, Iterable, Mapping, Sequence

from application.predictive_service import (
    PredictiveCacheSnapshot,
    PredictiveSnapshot,
    get_cache_stats,
)
from services.performance_metrics import MetricSummary, get_recent_metrics
from shared.settings import app_env
from shared.time_provider import TimeProvider
from shared.version import __build_signature__, __version__, get_version_info

"""Background diagnostics runner aggregating latency and environment health."""


_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)
_LOGGER.propagate = False

_LOG_ENV_NAME = "SYSTEM_DIAGNOSTICS_LOG_PATH"
_DEFAULT_LOG_PATH = Path("logs/system_diagnostics.log")


def _resolve_log_path() -> Path:
    raw_path = os.getenv(_LOG_ENV_NAME, str(_DEFAULT_LOG_PATH))
    path = Path(raw_path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        fallback = Path.cwd() / Path(raw_path).name
        fallback.parent.mkdir(parents=True, exist_ok=True)
        path = fallback
    return path


def _configure_logger() -> None:
    if any(getattr(handler, "_system_diagnostics_handler", False) for handler in _LOGGER.handlers):
        return
    handler = logging.FileHandler(_resolve_log_path(), encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    handler._system_diagnostics_handler = True  # type: ignore[attr-defined]
    _LOGGER.addHandler(handler)


_configure_logger()


@dataclass(frozen=True)
class EndpointLatencySnapshot:
    """Latency summary for an instrumented endpoint."""

    name: str
    average_ms: float | None
    baseline_ms: float | None
    degraded: bool
    samples: int
    last_ms: float | None
    last_timestamp: float | None

    @property
    def last_run_iso(self) -> str | None:
        if self.last_timestamp is None:
            return None
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_timestamp))


@dataclass(frozen=True)
class CacheHealthSnapshot:
    """State of the predictive cache exposed to the diagnostics UI."""

    hits: int
    misses: int
    hit_ratio: float
    last_updated: str
    ttl_hours: float | None
    remaining_ttl: float | None

    @classmethod
    def from_predictive(cls, snapshot: PredictiveSnapshot | PredictiveCacheSnapshot) -> "CacheHealthSnapshot":
        if isinstance(snapshot, PredictiveCacheSnapshot):
            base = snapshot.as_predictive_snapshot()
        else:
            base = snapshot
        return cls(
            hits=int(base.hits),
            misses=int(base.misses),
            hit_ratio=float(base.hit_ratio),
            last_updated=str(base.last_updated),
            ttl_hours=None if base.ttl_hours is None else float(base.ttl_hours),
            remaining_ttl=None if base.remaining_ttl is None else float(base.remaining_ttl),
        )

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hit_ratio,
            "last_updated": self.last_updated,
            "ttl_hours": self.ttl_hours,
            "remaining_ttl": self.remaining_ttl,
        }


@dataclass(frozen=True)
class FernetKeyStatus:
    """Validation details for a Fernet key defined in the environment."""

    name: str
    defined: bool
    valid: bool
    is_weak: bool | None
    fingerprint: str | None
    detail: str | None


@dataclass(frozen=True)
class EnvironmentStatus:
    """Minimal environment snapshot to expose via the diagnostics panel."""

    app_env: str | None
    timezone: str | None
    python_version: str
    platform: str


@dataclass(frozen=True)
class VersionMetadata:
    """Version metadata mirrored into diagnostics payloads."""

    version: str
    build_signature: str
    release_date: str | None
    codename: str | None
    stability: str | None

    def as_dict(self) -> dict[str, object | None]:
        return {
            "version": self.version,
            "build_signature": self.build_signature,
            "release_date": self.release_date,
            "codename": self.codename,
            "stability": self.stability,
        }


@dataclass(frozen=True)
class SystemDiagnosticsSnapshot:
    """Aggregate payload consumed by the diagnostics UI and logs."""

    generated_at: str
    endpoints: Sequence[EndpointLatencySnapshot]
    cache: CacheHealthSnapshot | None
    keys: Sequence[FernetKeyStatus]
    environment: EnvironmentStatus
    version: VersionMetadata

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "endpoints": [
                {
                    "name": entry.name,
                    "average_ms": entry.average_ms,
                    "baseline_ms": entry.baseline_ms,
                    "degraded": entry.degraded,
                    "samples": entry.samples,
                    "last_ms": entry.last_ms,
                    "last_timestamp": entry.last_timestamp,
                }
                for entry in self.endpoints
            ],
            "cache": None if self.cache is None else self.cache.as_dict(),
            "keys": [
                {
                    "name": entry.name,
                    "defined": entry.defined,
                    "valid": entry.valid,
                    "is_weak": entry.is_weak,
                    "fingerprint": entry.fingerprint,
                    "detail": entry.detail,
                }
                for entry in self.keys
            ],
            "environment": {
                "app_env": self.environment.app_env,
                "timezone": self.environment.timezone,
                "python_version": self.environment.python_version,
                "platform": self.environment.platform,
            },
            "version": self.version.as_dict(),
        }


@dataclass
class SystemDiagnosticsConfiguration:
    """Runtime configuration for the diagnostics scheduler."""

    interval_seconds: float = 300.0
    tracked_metrics: Sequence[str] = (
        "predictive_compute",
        "quotes_refresh",
        "apply_filters",
    )
    history_window: int = 20
    degradation_factor: float = 2.0


_HISTORY: dict[str, Deque[float]] = defaultdict(deque)
_STATE_LOCK = threading.Lock()
_LAST_SNAPSHOT: SystemDiagnosticsSnapshot | None = None
_CONFIG = SystemDiagnosticsConfiguration()


def _truncate_history(max_items: int) -> None:
    for history in _HISTORY.values():
        while len(history) > max_items:
            history.popleft()


def _resolve_generation_time(now: float | int | None) -> datetime:
    snapshot = TimeProvider.from_timestamp(now)
    if snapshot is not None:
        return snapshot.moment
    return TimeProvider.now_datetime()


def _safe_mean(values: Iterable[float]) -> float | None:
    values = tuple(values)
    if not values:
        return None
    return statistics.fmean(values)


def _record_history(metric: str, value: float | None) -> float | None:
    history = _HISTORY[metric]
    baseline = _safe_mean(history)
    if value is None:
        return baseline
    history.append(float(value))
    if len(history) > max(1, int(_CONFIG.history_window)):
        history.popleft()
    return baseline


def _decode_fernet_key(value: str) -> bytes:
    normalized = value.strip()
    if not normalized:
        raise ValueError("empty")
    padding = (-len(normalized)) % 4
    padded = normalized + ("=" * padding)
    decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
    if len(decoded) != 32:
        raise ValueError("invalid_length")
    return decoded


def _is_weak_key(decoded: bytes) -> bool:
    unique = {b for b in decoded}
    return len(unique) <= 4


def _fingerprint(value: str) -> str:
    return sha1(value.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]


def _collect_key_statuses() -> list[FernetKeyStatus]:
    names = ("FASTAPI_TOKENS_KEY", "IOL_TOKENS_KEY")
    statuses: list[FernetKeyStatus] = []
    decoded_map: dict[str, str] = {}

    for name in names:
        raw_value = os.getenv(name)
        if raw_value is None:
            statuses.append(
                FernetKeyStatus(
                    name=name,
                    defined=False,
                    valid=False,
                    is_weak=None,
                    fingerprint=None,
                    detail="Variable no definida",
                )
            )
            continue
        try:
            decoded = _decode_fernet_key(raw_value)
        except Exception:
            statuses.append(
                FernetKeyStatus(
                    name=name,
                    defined=True,
                    valid=False,
                    is_weak=None,
                    fingerprint=None,
                    detail="Formato inválido — se espera clave Fernet base64 de 32 bytes",
                )
            )
            continue
        fingerprint = _fingerprint(raw_value.strip())
        decoded_map[name] = raw_value.strip()
        statuses.append(
            FernetKeyStatus(
                name=name,
                defined=True,
                valid=True,
                is_weak=_is_weak_key(decoded),
                fingerprint=fingerprint,
                detail=None,
            )
        )

    if len(decoded_map) == len(names):
        values = {decoded_map[name] for name in names}
        if len(values) == 1:
            for index, status in enumerate(statuses):
                statuses[index] = FernetKeyStatus(
                    name=status.name,
                    defined=status.defined,
                    valid=False,
                    is_weak=status.is_weak,
                    fingerprint=status.fingerprint,
                    detail="Las claves FASTAPI e IOL no pueden coincidir",
                )
    return statuses


def _build_environment_status() -> EnvironmentStatus:
    return EnvironmentStatus(
        app_env=app_env or os.getenv("APP_ENV"),
        timezone=os.getenv("TZ"),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        platform=platform.platform(),
    )


def _resolve_metric(name: str, metrics: Mapping[str, MetricSummary]) -> EndpointLatencySnapshot:
    summary = metrics.get(name)
    average = summary.average_ms if summary else None
    baseline = _record_history(name, average)
    degraded = False
    if (
        average is not None
        and baseline is not None
        and baseline > 0
        and average > baseline * max(1.0, float(_CONFIG.degradation_factor))
    ):
        degraded = True
    samples = summary.samples if summary else 0
    last_ms = summary.last_ms if summary else None
    last_timestamp = summary.last_timestamp if summary else None
    return EndpointLatencySnapshot(
        name=name,
        average_ms=None if average is None else float(average),
        baseline_ms=None if baseline is None else float(baseline),
        degraded=degraded,
        samples=samples,
        last_ms=None if last_ms is None else float(last_ms),
        last_timestamp=None if last_timestamp is None else float(last_timestamp),
    )


def _collect_metrics() -> dict[str, MetricSummary]:
    return {summary.name: summary for summary in get_recent_metrics()}


def _collect_cache_snapshot() -> CacheHealthSnapshot | None:
    try:
        snapshot = get_cache_stats()
    except Exception:
        return None
    if not isinstance(snapshot, PredictiveSnapshot):
        return None
    return CacheHealthSnapshot.from_predictive(snapshot)


def _log_snapshot(snapshot: SystemDiagnosticsSnapshot) -> None:
    payload = snapshot.as_dict()
    _LOGGER.info(json.dumps(payload, ensure_ascii=False))


def _build_version_metadata() -> VersionMetadata:
    data = get_version_info()
    return VersionMetadata(
        version=str(data.get("version", __version__)),
        build_signature=str(data.get("build_signature", __build_signature__)),
        release_date=data.get("release_date"),
        codename=data.get("codename"),
        stability=data.get("stability"),
    )


def run_system_diagnostics_once(now: float | None = None) -> SystemDiagnosticsSnapshot:
    """Execute a diagnostics cycle and persist the snapshot."""

    with _STATE_LOCK:
        moment = _resolve_generation_time(now)
        generated_at = moment.isoformat()
        metrics = _collect_metrics()
        endpoints = [_resolve_metric(name, metrics) for name in _CONFIG.tracked_metrics]
        cache_snapshot = _collect_cache_snapshot()
        key_statuses = _collect_key_statuses()
        environment = _build_environment_status()
        snapshot = SystemDiagnosticsSnapshot(
            generated_at=generated_at,
            endpoints=endpoints,
            cache=cache_snapshot,
            keys=key_statuses,
            environment=environment,
            version=_build_version_metadata(),
        )
        global _LAST_SNAPSHOT
        _LAST_SNAPSHOT = snapshot
    _log_snapshot(snapshot)
    return snapshot


class _SystemDiagnosticsScheduler:
    def __init__(self, configuration: SystemDiagnosticsConfiguration) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._configuration = configuration

    def _get_interval(self) -> float:
        with self._lock:
            return float(self._configuration.interval_seconds)

    def update_configuration(self, configuration: SystemDiagnosticsConfiguration) -> None:
        with self._lock:
            self._configuration = configuration

    def ensure_running(self) -> bool:
        interval = max(0.0, self._get_interval())
        if interval <= 0:
            _LOGGER.info(
                "system-diagnostics scheduler disabled interval=%.2fs",
                interval,
            )
            return False
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="system-diagnostics",
                daemon=True,
            )
            self._thread.start()
            _LOGGER.info(
                "system-diagnostics scheduler started interval=%.2fs",
                interval,
            )
            return True

    def _run_loop(self) -> None:
        while not self._stop_event.wait(max(60.0, self._get_interval())):
            try:
                run_system_diagnostics_once()
            except Exception:
                _LOGGER.exception("system-diagnostics cycle failed")


_SCHEDULER = _SystemDiagnosticsScheduler(_CONFIG)


def configure_system_diagnostics(configuration: SystemDiagnosticsConfiguration) -> None:
    """Apply configuration updates for the diagnostics scheduler."""

    global _CONFIG
    _CONFIG = configuration
    with _STATE_LOCK:
        _truncate_history(max(1, int(configuration.history_window)))
    _SCHEDULER.update_configuration(configuration)


def ensure_system_diagnostics_started() -> bool:
    """Start the diagnostics scheduler if it is not running."""

    return _SCHEDULER.ensure_running()


def get_system_diagnostics_snapshot() -> SystemDiagnosticsSnapshot:
    """Expose the most recent diagnostics snapshot."""

    with _STATE_LOCK:
        snapshot = _LAST_SNAPSHOT
    if snapshot is None:
        snapshot = run_system_diagnostics_once()
    return snapshot


__all__ = [
    "CacheHealthSnapshot",
    "EndpointLatencySnapshot",
    "EnvironmentStatus",
    "FernetKeyStatus",
    "SystemDiagnosticsConfiguration",
    "SystemDiagnosticsSnapshot",
    "VersionMetadata",
    "configure_system_diagnostics",
    "ensure_system_diagnostics_started",
    "get_system_diagnostics_snapshot",
    "run_system_diagnostics_once",
]
