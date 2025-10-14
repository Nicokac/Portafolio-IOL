"""Startup logging utilities for capturing initialization failures early."""

from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from shared.version import __version__

_LOGGER_NAME: Final[str] = "app.startup"
_LOG_PATH: Final[Path] = Path("logs") / "app_startup.log"


def _ensure_logger_configured() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(_LOG_PATH, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_startup_event(message: str) -> None:
    """Record a structured startup event with process and version context."""
    logger = _ensure_logger_configured()
    logger.info("%s | pid=%s | version=%s", message, os.getpid(), __version__)


def log_startup_exception(exc: Exception) -> None:
    """Persist an unexpected startup exception with full traceback."""
    logger = _ensure_logger_configured()
    logger.error(
        "Startup exception captured | pid=%s | version=%s | error=%s",
        os.getpid(),
        __version__,
        exc,
    )
    logger.error("Traceback:\n%s", traceback.format_exc())


def log_ui_total_load_metric(total_ms: float | int | None, *, timestamp: datetime | None = None) -> None:
    """Record the total UI load metric with contextual metadata."""

    logger = _ensure_logger_configured()
    ts = timestamp or datetime.now(timezone.utc)
    ts_utc = ts.astimezone(timezone.utc)
    payload = {
        "metric": "ui_total_load",
        "value_ms": None if total_ms is None else float(total_ms),
        "version": __version__,
        "timestamp": ts_utc.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    logger.info(json.dumps(payload, ensure_ascii=False))


__all__ = [
    "log_startup_event",
    "log_startup_exception",
    "log_ui_total_load_metric",
]
