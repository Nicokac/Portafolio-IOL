from __future__ import annotations

from pathlib import Path
import tomllib


def _read_version() -> str:
    project_file = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        with project_file.open("rb") as f:
            data = tomllib.load(f)
        return data.get("project", {}).get("version", "0.0.0")
    except Exception:
        return "0.0.0"


__version__ = _read_version()
