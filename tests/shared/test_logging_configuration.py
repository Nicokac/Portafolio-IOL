import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

from shared import config


@pytest.mark.parametrize("json_format", [False, True])
def test_configure_logging_sets_matplotlib_font_manager_level(json_format):
    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_handlers = root_logger.handlers[:]
    original_matplotlib_level = logging.getLogger("matplotlib.font_manager").level

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    try:
        config.configure_logging(level="INFO", json_format=json_format)
        assert logging.getLogger("matplotlib.font_manager").level == logging.WARNING
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        for handler in original_handlers:
            root_logger.addHandler(handler)

        root_logger.setLevel(original_level)
        logging.getLogger("matplotlib.font_manager").setLevel(original_matplotlib_level)


@pytest.mark.parametrize("json_format", [False, True])
def test_configure_logging_adds_rotating_file_handler(tmp_path, monkeypatch, json_format):
    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_handlers = root_logger.handlers[:]

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    monkeypatch.setattr(config, "BASE_DIR", tmp_path)

    try:
        config.configure_logging(level="INFO", json_format=json_format)
        file_handlers = [
            handler for handler in root_logger.handlers if isinstance(handler, config.DailyTimedRotatingFileHandler)
        ]
        assert len(file_handlers) == 1

        log_file = Path(file_handlers[0].baseFilename)
        assert log_file.parent == tmp_path
        today = datetime.now().strftime("%Y-%m-%d")
        assert log_file.name == f"analysis_{today}.log"

        root_logger.info("hello world")

        for handler in root_logger.handlers:
            flush = getattr(handler, "flush", None)
            if callable(flush):
                flush()

        contents = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert contents

        last_line = contents[-1]
        if json_format:
            record = json.loads(last_line)
            assert record["message"] == "hello world"
            assert record["level"] == "INFO"
        else:
            assert "hello world" in last_line
            assert " - INFO - " in last_line
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        for handler in original_handlers:
            root_logger.addHandler(handler)

        root_logger.setLevel(original_level)
