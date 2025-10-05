"""Utilities to cleanup rotated log files based on retention policy."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

LOG_RETENTION_DAYS = 7

logger = logging.getLogger(__name__)


def _iter_log_files(directory: Path) -> Iterable[Path]:
    for path in directory.glob("*.log"):
        if path.is_file():
            yield path


def cleanup_log_directory(directory: Path, *, now: float | None = None) -> list[Path]:
    """Remove log files older than :data:`LOG_RETENTION_DAYS` days.

    Parameters
    ----------
    directory:
        Target directory where rotated logs reside.
    now:
        Optional timestamp override useful for testing. Defaults to
        :func:`time.time` when omitted.
    """

    directory = Path(directory)
    if not directory.exists():
        return []

    reference = float(now if now is not None else time.time())
    retention_seconds = LOG_RETENTION_DAYS * 86400

    removed: list[Path] = []
    for log_file in _iter_log_files(directory):
        try:
            mtime = log_file.stat().st_mtime
        except OSError:  # pragma: no cover - transient FS race
            logger.debug("No se pudo obtener mtime de %s", log_file, exc_info=True)
            continue
        age = reference - mtime
        if age > retention_seconds:
            try:
                log_file.unlink()
            except OSError:  # pragma: no cover - deletion failure is logged
                logger.warning("No se pudo eliminar log antiguo %s", log_file, exc_info=True)
            else:
                removed.append(log_file)
    return removed


__all__ = ["LOG_RETENTION_DAYS", "cleanup_log_directory"]

