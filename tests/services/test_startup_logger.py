from __future__ import annotations
from pathlib import Path

from services import startup_logger


def test_startup_logger_records_import_error():
    log_path = Path("logs") / "app_startup.log"
    previous_contents = log_path.read_text() if log_path.exists() else None

    if log_path.exists():
        log_path.unlink()

    try:
        raise ImportError("No module named 'tests.missing_module'")
    except ImportError as exc:
        startup_logger.log_startup_exception(exc)

    try:
        assert log_path.exists(), "Startup log file should be created on exception"
        contents = log_path.read_text()
        assert "tests.missing_module" in contents
        assert "Startup exception captured" in contents
    finally:
        if previous_contents is not None:
            log_path.write_text(previous_contents)
        else:
            if log_path.exists():
                log_path.unlink()
            parent = log_path.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
