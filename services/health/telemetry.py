"""Telemetry helpers for health metrics modules."""

from __future__ import annotations

import logging
from typing import Any, Mapping

_analysis_logger = logging.getLogger("analysis")


def log_analysis_event(event: str, latest: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    """Emit a structured analysis log entry when metrics are updated."""
    if not metrics:
        return

    _analysis_logger.info(
        "%s updated",
        event,
        extra={
            "analysis": {
                "event": event,
                "latest": dict(latest),
                "metrics": dict(metrics),
            }
        },
    )


__all__ = ["log_analysis_event"]
