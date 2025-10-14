from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from importlib import reload
from pathlib import Path

import pytest

from shared import version
from services import startup_logger


def test_ui_total_load_metric_log_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reload(startup_logger)
    log_path = tmp_path / "app_startup.log"
    logger = logging.getLogger("app.startup")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    monkeypatch.setattr(startup_logger, "_LOG_PATH", log_path, raising=False)

    fixed_timestamp = datetime(2025, 10, 13, 4, 31, 2, tzinfo=timezone.utc)
    startup_logger.log_ui_total_load_metric(8532, timestamp=fixed_timestamp)

    assert log_path.exists()
    contents = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert contents, "Startup log should contain at least one entry"
    payload = json.loads(contents[-1].split(" [INFO] ", 1)[-1])
    assert payload["metric"] == "ui_total_load"
    assert payload["value_ms"] == 8532.0
    assert payload["version"] == version.__version__
    assert payload["timestamp"] == "2025-10-13T04:31:02Z"
