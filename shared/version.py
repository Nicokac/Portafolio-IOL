from __future__ import annotations

from pathlib import Path
import tomllib


DEFAULT_VERSION = "0.3.4"
PROJECT_FILE = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _read_version() -> str:
    try:
        with PROJECT_FILE.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        return DEFAULT_VERSION

    version = data.get("project", {}).get("version", DEFAULT_VERSION)
    return version if isinstance(version, str) and version else DEFAULT_VERSION


__version__ = _read_version()
