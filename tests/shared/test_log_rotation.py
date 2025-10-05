import logging
from datetime import datetime, timedelta

import pytest

from shared import config


@pytest.fixture()
def _isolated_root_logger():
    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_handlers = root_logger.handlers[:]

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    try:
        yield root_logger
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


def test_configure_logging_prunes_old_log_files(tmp_path, monkeypatch, _isolated_root_logger):
    retention_days = 3
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    monkeypatch.setattr(config.settings, "LOG_RETENTION_DAYS", retention_days, raising=False)

    today = datetime.now().date()
    all_dates = [today - timedelta(days=offset) for offset in range(retention_days + 3)]
    for current_date in all_dates:
        log_path = tmp_path / f"analysis_{current_date.strftime('%Y-%m-%d')}.log"
        log_path.write_text("legacy", encoding="utf-8")

    legacy_path = tmp_path / "analysis.log"
    legacy_path.write_text("legacy", encoding="utf-8")

    config.configure_logging(level="INFO", json_format=False)

    remaining_files = {path.name for path in tmp_path.glob("analysis_*.log")}

    expected_kept = {
        f"analysis_{(today - timedelta(days=offset)).strftime('%Y-%m-%d')}.log"
        for offset in range(retention_days)
    }
    assert expected_kept <= remaining_files

    removed_cutoff = today - timedelta(days=retention_days)
    assert f"analysis_{removed_cutoff.strftime('%Y-%m-%d')}.log" not in remaining_files
    assert not legacy_path.exists()

    # Logging still works and writes to today's file.
    _isolated_root_logger.info("rotation test")
    today_file = tmp_path / f"analysis_{today.strftime('%Y-%m-%d')}.log"
    assert today_file.exists()
    assert "rotation test" in today_file.read_text(encoding="utf-8")
