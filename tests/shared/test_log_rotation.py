import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.log_rotation import LOG_RETENTION_DAYS, cleanup_log_directory


def test_cleanup_log_directory_respects_retention(tmp_path):
    old_file = tmp_path / "old.log"
    recent_file = tmp_path / "recent.log"
    boundary_file = tmp_path / "boundary.log"

    old_file.write_text("old", encoding="utf-8")
    recent_file.write_text("recent", encoding="utf-8")
    boundary_file.write_text("boundary", encoding="utf-8")

    now = 1_700_000_000.0
    retention_seconds = LOG_RETENTION_DAYS * 86400

    os.utime(old_file, (now - retention_seconds - 10, now - retention_seconds - 10))
    os.utime(recent_file, (now - retention_seconds + 10, now - retention_seconds + 10))
    os.utime(boundary_file, (now - retention_seconds, now - retention_seconds))

    removed = cleanup_log_directory(tmp_path, now=now)

    assert not old_file.exists()
    assert recent_file.exists()
    assert boundary_file.exists()

    removed_paths = {path.resolve() for path in removed}
    assert removed_paths == {old_file.resolve()}

