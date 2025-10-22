"""Helpers to capture runtime environment snapshots for diagnostics."""

from __future__ import annotations

import importlib.metadata
import logging
import os
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from shared.time_provider import TimeProvider

logger = logging.getLogger(__name__)

_SENSITIVE_TOKENS = ("PASSWORD", "SECRET", "TOKEN", "KEY", "PASS", "CREDENTIAL")


def _mask_if_sensitive(key: str, value: Any) -> str:
    key_normalized = key.upper()
    if any(token in key_normalized for token in _SENSITIVE_TOKENS):
        return "***"
    return str(value)


def _normalise_environment(env: Mapping[str, Any]) -> dict[str, str]:
    """Return a sanitised mapping with predictable ordering."""

    sanitised: dict[str, str] = {}
    for raw_key, raw_value in env.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if not value:
            continue
        sanitised[key] = _mask_if_sensitive(key, value)
    return dict(sorted(sanitised.items()))


def _normalise_packages(packages: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Return a sorted list of packages with name and version."""

    normalised: list[dict[str, str]] = []
    for entry in packages:
        name_raw = entry.get("name") if isinstance(entry, Mapping) else None
        version_raw = entry.get("version") if isinstance(entry, Mapping) else None
        name = str(name_raw or "").strip()
        version = str(version_raw or "").strip()
        if not name:
            continue
        normalised.append({"name": name, "version": version or "unknown"})
    normalised.sort(key=lambda item: item["name"].lower())
    return normalised


def _installed_packages() -> list[dict[str, str]]:
    try:
        dists = importlib.metadata.distributions()
    except Exception:  # pragma: no cover - importlib metadata edge cases
        logger.debug("No se pudo listar paquetes instalados", exc_info=True)
        return []

    packages: list[dict[str, str]] = []
    for dist in dists:
        name = dist.metadata.get("Name") or dist.metadata.get("Summary") or dist.metadata.get("name")
        if not name:
            continue
        version = getattr(dist, "version", None) or dist.metadata.get("Version") or "unknown"
        packages.append({"name": str(name).strip(), "version": str(version).strip()})
    packages.sort(key=lambda item: item["name"].lower())
    return packages


def build_environment_snapshot(
    *,
    env: Mapping[str, Any] | None = None,
    packages: Sequence[Mapping[str, Any]] | None = None,
    include_installed_packages: bool = True,
    log: bool = True,
) -> dict[str, Any]:
    """Collect metadata about the runtime environment.

    Parameters
    ----------
    env:
        Optional mapping with environment variables to record. When omitted, the
        snapshot contains the sanitized subset of ``os.environ``.
    packages:
        Optional iterable with explicit packages to record. When omitted and
        ``include_installed_packages`` is ``True`` the helper gathers the
        installed distributions via :mod:`importlib.metadata`.
    include_installed_packages:
        Controls whether installed packages should be inspected when ``packages``
        is not provided. Useful for tests to keep the snapshot deterministic.
    log:
        When ``True`` the snapshot is emitted through :mod:`logging` using the
        ``environment.snapshot`` event.
    """

    timestamp = TimeProvider.now()
    epoch = time.time()

    env_source: Mapping[str, Any]
    if env is None:
        env_source = os.environ
    else:
        env_source = env
    environment = _normalise_environment(env_source)

    if packages is not None:
        packages_list = _normalise_packages(packages)
    elif include_installed_packages:
        packages_list = _installed_packages()
    else:
        packages_list = []

    snapshot = {
        "event": "environment.snapshot",
        "timestamp": timestamp,
        "ts": epoch,
        "runtime": {
            "python": {
                "version": sys.version,
                "implementation": platform.python_implementation(),
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "executable": sys.executable,
        },
        "environment": environment,
        "packages": packages_list,
    }

    if log:
        logger.info("environment.snapshot", extra={"analysis": snapshot})

    return snapshot


__all__ = ["build_environment_snapshot"]
